import rp


def install_video_save_compat(fallback_bitrate=8_000_000, default_backend="imageio"):
    if getattr(rp, "_gwf_video_save_compat_installed", False):
        return

    original_save_video_mp4 = rp.save_video_mp4

    def safe_save_video_mp4(*args, **kwargs):
        if kwargs.get("video_bitrate") == "max":
            kwargs["video_bitrate"] = fallback_bitrate
        kwargs.setdefault("backend", default_backend)
        return original_save_video_mp4(*args, **kwargs)

    rp.save_video_mp4 = safe_save_video_mp4
    rp._gwf_video_save_compat_installed = True
