#!/usr/bin/env python3
"""自动为文件夹里的视频生成中文内容备注,并写入视频元数据。

流程:ffmpeg 压缩 → 上传 Gemini(2.5 Flash-Lite)生成描述 → exiftool 写回元数据 → 记录 CSV。
用法:python tag_videos.py <视频文件夹> [--limit N] [--dry-run] [--force] [--sleep 秒]
"""

import argparse
import csv
import datetime
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

MODEL = "gemini-2.5-flash-lite"
TAG_PREFIX = "[AI] "  # 已处理标记:元数据描述以此开头的视频会被跳过
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}
EMBEDDABLE_EXTS = {".mp4", ".mov", ".m4v"}  # exiftool 无法写入 avi/mkv,这两类只记 CSV
MAX_CLIP_SECONDS = 600  # 超长视频只取前 10 分钟
CSV_NAME = "tagged_videos.csv"

# Flash-Lite 标准价,用于结束时估算花费(美元/百万 token)
PRICE_INPUT = 0.10
PRICE_OUTPUT = 0.40

PROMPT = (
    "分析这个视频,返回 JSON,包含两个字段:\n"
    "description:用一两句中文描述这个视频大概拍了什么——有哪些人物、什么场景、在做什么活动;"
    "如果有说话内容或音乐值得一提也可以提及。不要开场白、不要引号。\n"
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
            raw = str(data.get(key) or "")
            m = re.match(r"(\d{4}):(\d{2}):(\d{2})", raw)
            # QuickTime 未知日期常写成 0000:00:00,要排除
            if m and m.group(1) != "0000":
                date = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
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
    dst = tmp_dir / (src.stem + ".compressed.mp4")
    cmd = [
        "ffmpeg", "-nostdin", "-y", "-v", "error",
        "-i", str(src),
        "-t", str(MAX_CLIP_SECONDS),
        "-vf", "scale=-2:360,fps=2",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "32",
        "-c:a", "aac", "-b:a", "48k", "-ac", "1",
        "-movflags", "+faststart",
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg 压缩失败: {result.stderr.strip()[:300]}")
    return dst


def describe(client: genai.Client, clip: Path, retries: int = 4) -> tuple[str, str, int, int]:
    """上传压缩片段并生成描述。返回 (描述, 文件名短语, 输入token, 输出token)。"""
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
                    model=MODEL,
                    contents=[uploaded, PROMPT],
                    config=genai_types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=RESPONSE_SCHEMA,
                    ),
                )
                break
            except genai_errors.APIError as e:
                retryable = e.code in (429, 500, 502, 503, 504)
                if not retryable or attempt == retries:
                    raise
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


def append_csv(csv_path: Path, row: dict) -> None:
    fields = ["path", "new_path", "description", "status", "input_tokens", "output_tokens", "seconds"]
    new_file = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="自动为视频生成中文内容备注")
    parser.add_argument("folder", type=Path, help="视频所在文件夹(递归扫描)")
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 个(0 = 不限)")
    parser.add_argument("--dry-run", action="store_true", help="只生成描述并打印,不写入文件")
    parser.add_argument("--force", action="store_true", help="重新处理已有 [AI] 描述的视频")
    parser.add_argument("--no-rename", action="store_true", help="只写元数据,不改文件名")
    parser.add_argument("--sleep", type=float, default=0, help="每个视频之间的间隔秒数(免费额度限速时用)")
    args = parser.parse_args()

    if not args.folder.is_dir():
        print(f"错误:{args.folder} 不是文件夹", file=sys.stderr)
        return 1
    if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
        print("错误:请先设置 GEMINI_API_KEY 环境变量(在 aistudio.google.com 免费申请)", file=sys.stderr)
        return 1

    client = genai.Client()
    csv_path = args.folder / CSV_NAME
    videos = find_videos(args.folder)
    if args.limit:
        videos = videos[: args.limit]
    print(f"找到 {len(videos)} 个视频待处理")

    done = skipped = failed = 0
    total_in = total_out = 0

    with tempfile.TemporaryDirectory(prefix="video-tagger-") as tmp:
        tmp_dir = Path(tmp)
        for i, video in enumerate(videos, 1):
            rel = video.relative_to(args.folder)
            existing, shot_date = read_metadata(video)
            if existing and existing.startswith(TAG_PREFIX) and not args.force:
                print(f"[{i}/{len(videos)}] 跳过(已有描述): {rel}")
                skipped += 1
                continue

            start = time.monotonic()
            try:
                clip = compress(video, tmp_dir)
                desc, slug, tok_in, tok_out = describe(client, clip)
                clip.unlink(missing_ok=True)
                total_in += tok_in
                total_out += tok_out

                new_stem = f"{shot_date}-{sanitize_slug(slug)}"
                if args.dry_run:
                    new_name = f"{new_stem}{video.suffix.lower()}"
                    print(f"[{i}/{len(videos)}] (dry-run) {rel}\n    → {desc}\n    → 新文件名: {new_name}")
                else:
                    if video.suffix.lower() in EMBEDDABLE_EXTS:
                        embed_description(video, desc)
                        status = "ok"
                    else:
                        status = "csv-only"  # avi/mkv 无法嵌入,只记录到 CSV
                    new_rel = rel
                    if not args.no_rename:
                        target = unique_target(video, new_stem, video.suffix.lower())
                        if target != video:
                            video.rename(target)
                        new_rel = target.relative_to(args.folder)
                    append_csv(csv_path, {
                        "path": str(rel), "new_path": str(new_rel),
                        "description": desc, "status": status,
                        "input_tokens": tok_in, "output_tokens": tok_out,
                        "seconds": round(time.monotonic() - start, 1),
                    })
                    note = "(仅记入 CSV,该格式无法嵌入)" if status == "csv-only" else ""
                    print(f"[{i}/{len(videos)}] 完成{note}: {rel} → {new_rel}\n    → {desc}")
                done += 1
            except Exception as e:
                failed += 1
                print(f"[{i}/{len(videos)}] 失败: {rel} — {e}", file=sys.stderr)
                if not args.dry_run:
                    append_csv(csv_path, {
                        "path": str(rel), "new_path": str(rel),
                        "description": "", "status": f"error: {e}",
                        "input_tokens": 0, "output_tokens": 0,
                        "seconds": round(time.monotonic() - start, 1),
                    })
            if args.sleep:
                time.sleep(args.sleep)

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
