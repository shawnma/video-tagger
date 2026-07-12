# video-tagger

自动为文件夹里的视频生成一两句中文备注(大概拍了什么),写入视频元数据,并把文件改名为 `yyyy-mm-dd-内容短语.mp4` 的形式(如 `2024-05-03-孩子后院玩水枪.mov`)。

原理:用 ffmpeg 把视频压成 360p 低帧率的小文件 → 上传给 Gemini 2.5 Flash-Lite(能看画面、听声音)生成完整描述 + 文件名短语 → 用 exiftool 把描述写入视频元数据(`Keys:Description` 和 `QuickTime:Comment`)→ 按拍摄日期 + 短语改名,同时记录一份 `tagged_videos.csv` 索引(含新旧文件名对照)。

日期取视频元数据里的拍摄时间(`Keys:CreationDate` / `CreateDate`),取不到时用文件修改时间。

成本:约 $0.002/条(1000 条视频约 $2);Gemini 免费额度内可能一分钱不花。

## 准备

需要 `ffmpeg` 和 `exiftool`(`brew install ffmpeg exiftool`,你已装好),以及一个 Gemini API key(在 [aistudio.google.com](https://aistudio.google.com/apikey) 免费申请)。

```sh
pip install -r requirements.txt
export GEMINI_API_KEY="你的key"
```

## 用法

```sh
# 先拿 3 个视频试效果,只打印描述、不写文件
python tag_videos.py ~/我的视频 --limit 3 --dry-run

# 满意后正式跑(可随时 Ctrl-C 中断,重跑会自动跳过已处理的)
python tag_videos.py ~/我的视频

# 用免费额度被限速时,加个间隔
python tag_videos.py ~/我的视频 --sleep 5

# 重新生成已有描述的视频
python tag_videos.py ~/我的视频 --force

# 只写元数据、不改文件名
python tag_videos.py ~/我的视频 --no-rename
```

## 说明

- 描述以 `[AI] ` 开头写入元数据,脚本以此判断是否已处理,支持断点续传。
- 新文件名重名时自动加 `-2`、`-3` 后缀;文件名短语最长 30 字,非法字符会被清洗掉。
- 想找回原文件名,查 `tagged_videos.csv` 的 path → new_path 对照列。
- 查看写入结果:Finder 里选中视频 ⌘I(获取信息),或 `exiftool -Keys:Description 视频.mp4`。
- `.mp4 / .mov / .m4v` 会嵌入元数据;`.avi / .mkv` exiftool 无法写入,描述只记在 CSV 里。
- exiftool 只修改元数据区,不触碰音视频流;写入保留文件原修改时间。
- 超过 10 分钟的视频只分析前 10 分钟。
