#!/usr/bin/env python3
"""自动为文件夹里的视频生成中文内容备注,并写入视频元数据。

流程:ffmpeg 压缩 → 上传 Gemini(2.5 Flash-Lite)生成描述 → exiftool 写回元数据 → 记录 CSV。
用法:python tag_videos.py <视频文件夹> [--limit N] [--dry-run] [--force] [--sleep 秒]
"""

import argparse
import concurrent.futures
import csv
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

MODEL = "gemini-3.1-flash-lite"
TAG_PREFIX = "[AI] "  # 已处理标记:元数据描述以此开头的视频会被跳过
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}
EMBEDDABLE_EXTS = {".mp4", ".mov", ".m4v"}  # exiftool 无法写入 avi/mkv,这两类只记 CSV
MAX_CLIP_SECONDS = 600  # 超长视频只取前 10 分钟
CSV_NAME = "tagged_videos.csv"

# 3.1 Flash-Lite 标准价,用于结束时估算花费(美元/百万 token;音频 token 按 $0.50 计,此处不细分,估算略偏低)
PRICE_INPUT = 0.25
PRICE_OUTPUT = 1.50

class BillingError(RuntimeError):
    """账户额度耗尽等不可重试错误,应中止整个运行而不是逐个视频重试。"""


PROMPT = (
    "分析这个视频,返回 JSON,包含两个字段:\n"
    "description:用一两句中文描述这个视频大概拍了什么——有哪些人物、什么场景、在做什么活动;"
    "如果有说话内容或音乐值得一提也可以提及。"
    "特别注意:如果视频中出现了具体的人名、地名或场合名——比如司仪口播的新人姓名、横幅或蛋糕上的文字、"
    "字幕、对话中反复被称呼的名字——一定写进描述里(如\"张伟和李娜的婚礼\");听不清或没有就不要编造。"
    "不要开场白、不要引号。\n"
    "filename:一个适合做文件名的中文短语,概括视频内容,4-15 个字,"
    "只用汉字、字母、数字,不要标点和空格,例如:孩子后院玩水枪。"
)

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "description": {"type": "STRING"},
        "filename": {"type": "STRING"},
    },
    "required": ["description", "filename"],
}


def find_videos(root: Path) -> list[Path]:
    videos = [
        p for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS and not p.name.startswith("._")
    ]
    return videos


def _parse_shot_date(raw: str) -> str | None:
    """把 exiftool 的日期字符串转成本地日期 yyyy-mm-dd。

    Keys:CreationDate 自带时区偏移(本地时间),日期直接用;
    CreateDate/MediaCreateDate 无偏移,按 QuickTime 规范视为 UTC,转本地时区
    (否则傍晚拍的视频日期会多一天)。年份不合理(设备乱写)则放弃该字段。
    """
    m = re.match(
        r"(\d{4}):(\d{2}):(\d{2})[ T](\d{2}):(\d{2}):(\d{2})(?:\.\d+)?(Z|[+-]\d{2}:?\d{2})?",
        raw,
    )
    if not m:
        return None
    y = int(m.group(1))
    if not (1980 <= y <= datetime.date.today().year + 1):
        return None
    if m.group(7):  # 带时区偏移 → 已是本地时间
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    dt = datetime.datetime(
        y, int(m.group(2)), int(m.group(3)),
        int(m.group(4)), int(m.group(5)), int(m.group(6)),
        tzinfo=datetime.timezone.utc,
    )
    return dt.astimezone().date().isoformat()


def read_metadata(path: Path) -> tuple[str | None, str | None]:
    """返回 (已有描述, 拍摄日期 yyyy-mm-dd)。日期取不到时退回文件修改时间。"""
    out = subprocess.run(
        ["exiftool", "-j", "-Keys:Description", "-QuickTime:Comment",
         "-Keys:CreationDate", "-QuickTime:CreateDate", "-MediaCreateDate", str(path)],
        capture_output=True, text=True,
    )
    desc = None
    date = None
    if out.returncode == 0 and out.stdout.strip():
        try:
            data = json.loads(out.stdout)[0]
        except (json.JSONDecodeError, IndexError):
            data = {}
        desc = data.get("Description") or data.get("Comment")
        for key in ("CreationDate", "CreateDate", "MediaCreateDate"):
            date = _parse_shot_date(str(data.get(key) or ""))
            if date:
                break
    if date is None:
        date = datetime.date.fromtimestamp(path.stat().st_mtime).isoformat()
    return desc, date


def sanitize_slug(slug: str) -> str:
    """把模型给的短语清洗成安全的文件名片段。"""
    slug = re.sub(r"\s+", "-", slug.strip())
    slug = re.sub(r"[^\w一-鿿-]", "", slug)  # 只留字母数字下划线、汉字、连字符
    slug = re.sub(r"-{2,}", "-", slug).strip("-_")
    return slug[:30] or "视频"


def unique_target(current: Path, stem: str, ext: str) -> Path:
    """在 current 所在目录里为 stem+ext 找一个不冲突的路径;current 自己不算冲突。"""
    directory = current.parent
    target = directory / f"{stem}{ext}"
    n = 2
    while target.exists() and not target.samefile(current):
        target = directory / f"{stem}-{n}{ext}"
        n += 1
    return target


def compress(src: Path, tmp_dir: Path) -> Path:
    # 临时文件名必须纯 ASCII:上传 SDK 会把文件名放进 HTTP 头,中文名会报编码错
    safe = hashlib.md5(str(src).encode()).hexdigest()[:16]
    dst = tmp_dir / f"clip-{safe}.mp4"
    common = [
        "ffmpeg", "-nostdin", "-y", "-v", "error",
        "-i", str(src), "-t", str(MAX_CLIP_SECONDS),
    ]
    tail = [
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "32",
        "-c:a", "aac", "-b:a", "48k", "-ac", "1",
        "-movflags", "+faststart", str(dst),
    ]
    result = subprocess.run(common + ["-vf", "scale=-2:360,fps=2"] + tail,
                            capture_output=True, text=True)
    if result.returncode != 0:
        # ffmpeg 8 的 scale 滤镜对个别文件报 "Impossible to convert between formats";
        # 降级为不缩放、仅降帧率的纯转码(这类文件通常分辨率本来就不高)
        result = subprocess.run(common + ["-r", "2"] + tail,
                                capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg 压缩失败: {result.stderr.strip()[:300]}")
    return dst


def describe(client: genai.Client, clip: Path, model: str = MODEL, retries: int = 4,
             people: str = "", shot_date: str = "") -> tuple[str, str, int, int]:
    """上传压缩片段并生成描述。返回 (描述, 文件名短语, 输入token, 输出token)。"""
    prompt = PROMPT
    if people:
        date_line = f"这个视频拍摄于 {shot_date}。\n" if shot_date else ""
        prompt = (
            f"{date_line}"
            f"以下是这个家庭视频库的人物名册,供辨认人物使用:\n\n{people}\n\n"
            "辨认规则:\n"
            "1. 以上面给出的拍摄日期为准(不要自己猜年份),用它减去出生日期算出每个孩子当时的年龄,"
            "只有年龄和画面中人物外貌相符才能认定是那个人;出生日期晚于拍摄日期的人绝不可能出现。\n"
            "2. 描述和文件名里统一使用名册中每人的第一个名字(主名),不要混用小名或别名。\n"
            "3. 有把握才写名字,拿不准就用泛称,不要猜。\n\n" + PROMPT
        )
    uploaded = client.files.upload(file=str(clip))
    try:
        deadline = time.monotonic() + 300
        while uploaded.state and uploaded.state.name == "PROCESSING":
            if time.monotonic() > deadline:
                raise RuntimeError("Gemini 文件处理超时")
            time.sleep(2)
            uploaded = client.files.get(name=uploaded.name)
        if uploaded.state and uploaded.state.name == "FAILED":
            raise RuntimeError("Gemini 无法处理该视频文件")

        delay = 5.0
        for attempt in range(retries + 1):
            try:
                resp = client.models.generate_content(
                    model=model,
                    contents=[uploaded, prompt],
                    config=genai_types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=RESPONSE_SCHEMA,
                    ),
                )
            except genai_errors.APIError as e:
                msg = str(e)
                if e.code == 429 and ("credits are depleted" in msg or "billing" in msg.lower()):
                    raise BillingError(msg)
                retryable = e.code in (429, 500, 502, 503, 504)
                if not retryable or attempt == retries:
                    raise
                time.sleep(delay)
                delay *= 2
                continue
            if (resp.text or "").strip():
                break
            # 空响应是偶发的服务端抖动,同样重试
            reason = resp.candidates[0].finish_reason if resp.candidates else None
            if attempt == retries:
                raise RuntimeError(f"模型多次返回空响应 (finish_reason={reason})")
            time.sleep(delay)
            delay *= 2

        try:
            data = json.loads(resp.text or "")
        except json.JSONDecodeError:
            raise RuntimeError(f"模型返回的不是有效 JSON: {(resp.text or '')[:100]!r}")
        desc = str(data.get("description", "")).strip()
        slug = str(data.get("filename", "")).strip()
        if not desc:
            raise RuntimeError("模型返回了空描述")
        usage = resp.usage_metadata
        return (
            desc,
            slug,
            usage.prompt_token_count or 0 if usage else 0,
            usage.candidates_token_count or 0 if usage else 0,
        )
    finally:
        try:
            client.files.delete(name=uploaded.name)
        except Exception:
            pass  # 48 小时后会自动过期,删除失败不影响结果


def embed_description(path: Path, description: str) -> None:
    tagged = TAG_PREFIX + description
    cmd = [
        "exiftool", "-m", "-P", "-overwrite_original",
        f"-Keys:Description={tagged}",
        f"-QuickTime:Comment={tagged}",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"exiftool 写入失败: {result.stderr.strip()[:300]}")


def set_finder_comment(path: Path, comment: str) -> None:
    """写 Finder 注释(⌘I 可见、Spotlight 可搜)。失败只警告,不影响主流程。"""
    result = subprocess.run(
        [
            "osascript",
            "-e", "on run argv",
            "-e", 'tell application "Finder" to set comment of '
                  "(POSIX file (item 1 of argv) as alias) to (item 2 of argv)",
            "-e", "end run",
            str(path), comment,
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"    警告:Finder 注释写入失败({result.stderr.strip()[:120]})", file=sys.stderr)


def append_csv(csv_path: Path, row: dict) -> None:
    fields = ["path", "new_path", "description", "status", "input_tokens", "output_tokens", "seconds"]
    new_file = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


_rename_lock = threading.Lock()  # 并行时防止两个视频抢同一个目标文件名


def process_one(client: genai.Client, args, folder: Path, tmp_dir: Path, video: Path) -> dict:
    """处理单个视频,供线程池调用。返回结果字典,kind ∈ skipped/dry/done。"""
    rel = video.relative_to(folder)
    existing, shot_date = read_metadata(video)
    if existing and existing.startswith(TAG_PREFIX) and not args.force:
        return {"kind": "skipped", "rel": rel}

    start = time.monotonic()
    clip = compress(video, tmp_dir)
    try:
        if args.sleep:
            time.sleep(args.sleep)
        desc, slug, tok_in, tok_out = describe(
            client, clip, model=args.model,
            people=args.people_text, shot_date=shot_date,
        )
    finally:
        clip.unlink(missing_ok=True)

    new_stem = f"{shot_date}-{sanitize_slug(slug)}"
    if args.dry_run:
        return {
            "kind": "dry", "rel": rel, "desc": desc,
            "new_name": f"{new_stem}{video.suffix.lower()}",
            "tok_in": tok_in, "tok_out": tok_out,
        }

    if video.suffix.lower() in EMBEDDABLE_EXTS:
        embed_description(video, desc)
        status = "ok"
    else:
        status = "csv-only"  # avi/mkv 无法嵌入,只记录到 CSV
    new_rel = rel
    final_path = video
    if not args.no_rename:
        with _rename_lock:
            target = unique_target(video, new_stem, video.suffix.lower())
            if target != video:
                video.rename(target)
        final_path = target
        new_rel = target.relative_to(folder)
    set_finder_comment(final_path, TAG_PREFIX + desc)
    return {
        "kind": "done", "rel": rel, "new_rel": new_rel, "desc": desc,
        "status": status, "tok_in": tok_in, "tok_out": tok_out,
        "seconds": round(time.monotonic() - start, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="自动为视频生成中文内容备注")
    parser.add_argument("folder", type=Path, help="视频所在文件夹(递归扫描)")
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 个(0 = 不限)")
    parser.add_argument("--dry-run", action="store_true", help="只生成描述并打印,不写入文件")
    parser.add_argument("--force", action="store_true", help="重新处理已有 [AI] 描述的视频")
    parser.add_argument("--no-rename", action="store_true", help="只写元数据,不改文件名")
    parser.add_argument("--model", default=MODEL, help=f"Gemini 模型名(默认 {MODEL})")
    parser.add_argument("--workers", type=int, default=4, help="并行处理数(默认 4)")
    parser.add_argument("--people", type=Path, default=None,
                        help="人物名册文本文件,帮助模型认出家庭成员(样例见 people.example.txt)")
    parser.add_argument("--files", type=Path, default=None,
                        help="只处理清单文件里列出的视频(每行一个路径,绝对路径或相对 folder)")
    parser.add_argument("--sleep", type=float, default=0, help="每个视频之间的间隔秒数(免费额度限速时用)")
    args = parser.parse_args()

    if not args.folder.is_dir():
        print(f"错误:{args.folder} 不是文件夹", file=sys.stderr)
        return 1
    if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
        print("错误:请先设置 GEMINI_API_KEY 环境变量(在 aistudio.google.com 免费申请)", file=sys.stderr)
        return 1

    args.people_text = ""
    if args.people:
        if not args.people.is_file():
            print(f"错误:名册文件不存在: {args.people}", file=sys.stderr)
            return 1
        args.people_text = args.people.read_text(encoding="utf-8").strip()

    client = genai.Client()
    csv_path = args.folder / CSV_NAME
    if args.files:
        videos = []
        for line in args.files.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = Path(line)
            if not p.is_absolute():
                p = args.folder / p
            if p.is_file():
                videos.append(p)
            else:
                print(f"警告:清单中的文件不存在,跳过: {line}", file=sys.stderr)
    else:
        videos = find_videos(args.folder)
    if args.limit:
        videos = videos[: args.limit]
    print(f"找到 {len(videos)} 个视频待处理")

    done = skipped = failed = 0
    total_in = total_out = 0

    with tempfile.TemporaryDirectory(prefix="video-tagger-") as tmp, \
         concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        tmp_dir = Path(tmp)
        futures = {
            pool.submit(process_one, client, args, args.folder, tmp_dir, v): v
            for v in videos
        }
        n = 0
        aborted = False
        for fut in concurrent.futures.as_completed(futures):
            video = futures[fut]
            rel = video.relative_to(args.folder)
            n += 1
            try:
                r = fut.result()
            except BillingError as e:
                if not aborted:
                    aborted = True
                    print(f"\n中止:API 额度已耗尽,请到 https://ai.studio/projects 处理账单后重跑。\n{e}", file=sys.stderr)
                    for pending in futures:
                        pending.cancel()
                failed += 1
                continue
            except concurrent.futures.CancelledError:
                continue
            except Exception as e:
                failed += 1
                print(f"[{n}/{len(videos)}] 失败: {rel} — {e}", file=sys.stderr)
                if not args.dry_run:
                    append_csv(csv_path, {
                        "path": str(rel), "new_path": str(rel),
                        "description": "", "status": f"error: {e}",
                        "input_tokens": 0, "output_tokens": 0, "seconds": 0,
                    })
                continue

            if r["kind"] == "skipped":
                skipped += 1
                print(f"[{n}/{len(videos)}] 跳过(已有描述): {r['rel']}")
            elif r["kind"] == "dry":
                done += 1
                total_in += r["tok_in"]
                total_out += r["tok_out"]
                print(f"[{n}/{len(videos)}] (dry-run) {r['rel']}\n    → {r['desc']}\n    → 新文件名: {r['new_name']}")
            else:
                done += 1
                total_in += r["tok_in"]
                total_out += r["tok_out"]
                append_csv(csv_path, {
                    "path": str(r["rel"]), "new_path": str(r["new_rel"]),
                    "description": r["desc"], "status": r["status"],
                    "input_tokens": r["tok_in"], "output_tokens": r["tok_out"],
                    "seconds": r["seconds"],
                })
                note = "(仅记入 CSV,该格式无法嵌入)" if r["status"] == "csv-only" else ""
                print(f"[{n}/{len(videos)}] 完成{note}: {r['rel']} → {r['new_rel']}\n    → {r['desc']}")

    cost = total_in / 1e6 * PRICE_INPUT + total_out / 1e6 * PRICE_OUTPUT
    print(
        f"\n汇总:成功 {done},跳过 {skipped},失败 {failed}"
        f" | token: 输入 {total_in:,} / 输出 {total_out:,}"
        f" | 估算花费 ≈ ${cost:.4f}(免费额度内则为 $0)"
    )
    if not args.dry_run:
        print(f"明细索引:{csv_path}")
    return 1 if failed and not done else 0


if __name__ == "__main__":
    sys.exit(main())
