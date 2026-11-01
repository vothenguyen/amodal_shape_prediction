"""
===================================================================================
HỆ THỐNG LOGGING & TELEMETRY BỀN VỮNG CHO HUẤN LUYỆN D2SA / QUEUE SERVER
===================================================================================
Module đảm bảo không mất bất kỳ kết quả / metric nào khi chạy trên GPU server
dùng cơ chế hàng đợi (queue manager) hoặc khi bị ngắt đột ngột (kill, quota, crash).

Tính năng chính:
1. Ghi đồng thời 2 file log theo mốc thời gian {YYYYMMDD}_{HHMMSS}:
   - File .log (human-readable, đầy đủ timestamp, thiết bị, hyperparams, kết quả)
   - File .jsonl (machine-readable, mỗi dòng 1 JSON object, fsync ngay lập tức)
2. Immediate Flush & Disk Sync:
   - Dùng custom FlushFileHandler và os.fsync() ép dữ liệu ghi xuống đĩa ngay
   - Không bị mất dữ liệu nằm trong buffer khi tiến trình bị kill
3. Bắt lỗi toàn diện:
   - try/except bắt Exception và ghi đầy đủ traceback + epoch đang chạy dở
   - try/finally ghi tổng kết thời gian chạy, checkpoint cuối, trạng thái
4. Xử lý tín hiệu SIGTERM:
   - Đăng ký signal handler để phát hiện cảnh báo kill từ queue manager
   - Cố gắng lưu checkpoint khẩn cấp và flush log trước khi thoát
5. Tương thích đa nền tảng (SafeStreamHandler):
   - Không bao giờ crash bởi UnicodeEncodeError trên Windows (cp1252) hay terminal hạn chế.
===================================================================================
"""

import os
import sys
import time
import json
import logging
import platform
import subprocess
import traceback
import signal
from datetime import datetime

# Cấu hình UTF-8 cho Windows console nếu có thể
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


class FlushFileHandler(logging.FileHandler):
    """
    FileHandler tự động flush và fsync sau mỗi lần ghi log,
    đảm bảo dữ liệu được ghi xuống đĩa cứng ngay lập tức.
    """
    def emit(self, record):
        super().emit(record)
        self.flush()
        try:
            if hasattr(self.stream, 'fileno'):
                os.fsync(self.stream.fileno())
        except Exception:
            pass


class SafeStreamHandler(logging.StreamHandler):
    """
    StreamHandler an toàn tuyệt đối, không bao giờ crash vì UnicodeEncodeError
    ngay cả khi terminal chạy cp1252, ASCII hoặc môi trường không hỗ trợ UTF-8.
    """
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            try:
                stream.write(msg + self.terminator)
            except UnicodeEncodeError:
                encoding = getattr(stream, 'encoding', None) or 'utf-8'
                safe_msg = msg.encode(encoding, errors='replace').decode(encoding, errors='replace')
                stream.write(safe_msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)


def get_git_commit_hash(project_root=None):
    """
    Lấy mã commit hash hiện tại của repository Git.
    Kiểm tra chính xác thư mục dự án có phải là git repository hay không
    (tránh trường hợp thư mục cha vô tình có .git).
    """
    if project_root is None:
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        if "other_config" in curr_dir:
            project_root = os.path.abspath(os.path.join(curr_dir, "..", "..", ".."))
        else:
            project_root = os.path.abspath(os.path.join(curr_dir, ".."))

    # 1. Kiểm tra trực tiếp thư mục .git trong project root
    git_dir = os.path.join(project_root, ".git")
    if not os.path.exists(git_dir):
        return "no-git-repo (extracted from zip archive)"

    try:
        # 2. Kiểm tra git toplevel có khớp project root không
        top_res = subprocess.run(
            ["git", "-C", project_root, "rev-parse", "--show-toplevel"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=3,
            check=True
        )
        git_toplevel = os.path.abspath(top_res.stdout.strip())
        if os.path.normpath(git_toplevel).lower() != os.path.normpath(project_root).lower():
            return "no-git-repo (extracted from zip archive)"

        # 3. Lấy commit hash
        hash_res = subprocess.run(
            ["git", "-C", project_root, "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=3,
            check=True
        )
        return hash_res.stdout.strip()
    except Exception:
        return "no-git-repo (git command failed or not installed)"


def get_environment_info():
    """Thu thập thông tin phần cứng, PyTorch, CUDA và hệ điều hành."""
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "None (CPU)"
        device_count = torch.cuda.device_count() if cuda_avail else 0
        cuda_ver = torch.version.cuda if cuda_avail else "N/A"
        pytorch_ver = torch.__version__
    except ImportError:
        cuda_avail = False
        gpu_name = "PyTorch not installed"
        device_count = 0
        cuda_ver = "N/A"
        pytorch_ver = "N/A"

    info = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": platform.node(),
        "pytorch_version": pytorch_ver,
        "cuda_available": cuda_avail,
        "cuda_version": cuda_ver,
        "device_count": device_count,
        "gpu_name": gpu_name,
    }
    return info


class TrainLogger:
    """
    Trình quản lý log cho quá trình huấn luyện.
    """
    def __init__(self, config_name, log_dir, run_timestamp=None):
        """
        Khởi tạo hệ thống logging cho một lượt huấn luyện.
        
        Args:
            config_name (str): Tên định danh cấu hình (vd: Row 1, Row 4 Main).
            log_dir (str): Thư mục lưu trữ log.
            run_timestamp (str, optional): Chuỗi timestamp {YYYYMMDD}_{HHMMSS}.
        """
        self.config_name = config_name
        self.log_dir = os.path.normpath(os.path.abspath(log_dir))
        os.makedirs(self.log_dir, exist_ok=True)

        if run_timestamp is None:
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            self.timestamp = run_timestamp

        self.log_path = os.path.join(self.log_dir, f"{self.timestamp}_train.log")
        self.jsonl_path = os.path.join(self.log_dir, f"{self.timestamp}_metrics.jsonl")

        # Thiết lập logger chuẩn
        self.logger_name = f"TrainLogger_{config_name}_{self.timestamp}".replace(" ", "_")
        self.logger = logging.getLogger(self.logger_name)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        # Xóa các handler cũ nếu tồn tại
        for h in list(self.logger.handlers):
            self.logger.removeHandler(h)

        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # 1) FileHandler có flush + fsync
        self.file_handler = FlushFileHandler(self.log_path, encoding="utf-8")
        self.file_handler.setLevel(logging.INFO)
        self.file_handler.setFormatter(formatter)
        self.logger.addHandler(self.file_handler)

        # 2) SafeStreamHandler xuất terminal an toàn
        self.stream_handler = SafeStreamHandler(sys.stdout)
        self.stream_handler.setLevel(logging.INFO)
        self.stream_handler.setFormatter(formatter)
        self.logger.addHandler(self.stream_handler)

        # File handle cho JSONL
        self.jsonl_file = open(self.jsonl_path, "a", encoding="utf-8")

        # State tracking
        self.start_time = time.time()
        self.current_epoch = 0
        self.last_completed_epoch = 0
        self.last_checkpoint_path = None
        self.status = "RUNNING"
        self.is_closed = False
        self.model_ref = None
        self.checkpoint_dir = None

    def _write_jsonl(self, data_dict):
        """Ghi 1 dòng JSON và fsync xuống đĩa ngay lập tức."""
        if self.jsonl_file and not self.jsonl_file.closed:
            line = json.dumps(data_dict, ensure_ascii=False)
            self.jsonl_file.write(line + "\n")
            self.jsonl_file.flush()
            try:
                os.fsync(self.jsonl_file.fileno())
            except Exception:
                pass

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def exception(self, msg):
        self.logger.exception(msg)

    def log_start(self, args, extra_info=None):
        """
        Ghi toàn bộ thông tin cấu hình, hyperparameter và môi trường khi bắt đầu chạy.
        """
        env_info = get_environment_info()
        git_hash = get_git_commit_hash()
        hparams = vars(args) if hasattr(args, "__dict__") else dict(args)

        bs = hparams.get("batch_size", "N/A")
        accum = hparams.get("accumulation_steps", 1)
        effective_bs = bs * accum if isinstance(bs, int) and isinstance(accum, int) else "N/A"
        start_dt = datetime.now()

        header = [
            "=" * 78,
            f">>> BẮT ĐẦU HUẤN LUYỆN: {self.config_name}",
            f"    Thời gian bắt đầu: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}",
            f"    Git commit hash:   {git_hash}",
            f"    Môi trường phần cứng & thư viện:",
            f"      - GPU:             {env_info['gpu_name']} (Số GPU: {env_info['device_count']})",
            f"      - CUDA / PyTorch:  {env_info['cuda_version']} / {env_info['pytorch_version']}",
            f"      - Hostname / OS:   {env_info['hostname']} / {env_info['platform']}",
            f"      - Python:          {env_info['python_version']}",
        ]

        if extra_info:
            header.append("    Kiến trúc & Tham số mô hình:")
            for ek, ev in extra_info.items():
                header.append(f"      - {ek}: {ev}")

        header.extend([
            f"    Hyperparameters:",
            f"      - Batch Size:      {bs} (Accumulation: {accum} -> Effective: {effective_bs})",
            f"      - Epochs:          {hparams.get('epochs', 'N/A')} (Resume: {hparams.get('resume_epoch', 0)})",
            f"      - Learning Rate:   {hparams.get('lr', 'N/A')}",
            f"      - Optimizer:       AdamW | Scheduler: CosineAnnealingLR",
            f"      - Occlusion Wgt:   {hparams.get('occlusion_weight', 'N/A')}",
            f"      - Num Classes:     {hparams.get('num_classes', 'N/A')}",
            f"      - Resize:          {hparams.get('resize', 'N/A')}",
            f"      - Workers:         {hparams.get('num_workers', 'N/A')}",
            f"      - Subset Size:     {hparams.get('subset_size', 'Toàn bộ')}",
            f"    Đường dẫn lưu trữ & dữ liệu:",
            f"      - Image Dir:       {hparams.get('img_dir', 'N/A')}",
            f"      - Ann Files:       {hparams.get('ann_files', 'N/A')}",
            f"      - Checkpoint Dir:  {hparams.get('checkpoint_dir', 'N/A')}",
            f"      - Log File (.log): {self.log_path}",
            f"      - Metrics (.jsonl):{self.jsonl_path}",
            "=" * 78,
        ])
        for line in header:
            self.logger.info(line)

        # Ghi metadata vào JSONL
        meta_record = {
            "type": "metadata",
            "config_name": self.config_name,
            "timestamp": start_dt.isoformat(),
            "git_commit": git_hash,
            "environment": env_info,
            "hyperparameters": {
                k: (list(v) if isinstance(v, (list, tuple)) else str(v) if not isinstance(v, (int, float, bool, type(None))) else v)
                for k, v in hparams.items()
            },
            "effective_batch_size": effective_bs,
            "log_path": self.log_path,
            "jsonl_path": self.jsonl_path
        }
        if extra_info:
            meta_record["extra_info"] = extra_info
        self._write_jsonl(meta_record)

    def set_current_epoch(self, epoch):
        """Cập nhật epoch đang tiến hành."""
        self.current_epoch = epoch

    def log_epoch(self, epoch, total_epochs, train_loss, duration_sec, lr, checkpoint_path=None, val_metrics=None, sub_losses=None, train_pos_pct=None):
        """
        Ghi nhận kết quả của 1 epoch hoàn thành. Ghi và fsync ngay lập tức.
        """
        self.last_completed_epoch = epoch
        if checkpoint_path:
            self.last_checkpoint_path = checkpoint_path

        msg = f"[EPOCH {epoch}/{total_epochs}] Loss: {train_loss:.4f} | Time: {duration_sec:.2f}s | LR: {lr:.2e}"
        if train_pos_pct is not None:
            msg += f" | PosPixels: {train_pos_pct:.2f}%"
        if sub_losses:
            loss_str = ", ".join([f"{k}: {v:.4f}" for k, v in sub_losses.items()])
            msg += f" | ({loss_str})"
        if val_metrics:
            val_items = []
            for k, v in val_metrics.items():
                if k in ("pos_pixel_pct", "pos_pixels"):
                    val_items.append(f"{k}: {v:.2f}%" if isinstance(v, (int, float)) else f"{k}: {v}")
                elif isinstance(v, float):
                    val_items.append(f"{k}: {v:.4f}")
                else:
                    val_items.append(f"{k}: {v}")
            val_str = ", ".join(val_items)
            msg += f" | Val: [{val_str}]"
        if checkpoint_path:
            msg += f" | Checkpoint: {checkpoint_path}"

        self.logger.info(msg)

        epoch_record = {
            "type": "epoch",
            "epoch": epoch,
            "total_epochs": total_epochs,
            "train_loss": round(float(train_loss), 6),
            "train_pos_pixel_pct": round(float(train_pos_pct), 2) if train_pos_pct is not None else None,
            "duration_sec": round(float(duration_sec), 2),
            "lr": float(lr),
            "checkpoint_path": checkpoint_path,
            "val_metrics": val_metrics,
            "sub_losses": sub_losses,
            "timestamp": datetime.now().isoformat()
        }
        self._write_jsonl(epoch_record)

    def log_error(self, exc, epoch=None):
        """
        Ghi lại exception và traceback đầy đủ khi xảy ra lỗi/crash.
        """
        self.status = "FAILED"
        err_epoch = epoch if epoch is not None else self.current_epoch
        tb_str = traceback.format_exc()

        err_lines = [
            "!" * 78,
            f"[ERROR] PHÁT HIỆN LỖI (EXCEPTION) TẠI EPOCH {err_epoch}!",
            f"Loại lỗi: {type(exc).__name__}: {str(exc)}",
            f"Dấu vết Stack Trace:",
            tb_str.strip(),
            f"[RESUME NOTE] Có thể tiếp tục chạy từ epoch {self.last_completed_epoch} bằng cờ --resume-epoch {self.last_completed_epoch}",
            "!" * 78,
        ]
        for line in err_lines:
            self.logger.error(line)

        err_record = {
            "type": "error",
            "epoch_in_progress": err_epoch,
            "last_completed_epoch": self.last_completed_epoch,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": tb_str,
            "timestamp": datetime.now().isoformat()
        }
        self._write_jsonl(err_record)

    def log_end(self, final_metrics=None):
        """
        Ghi lại tổng kết kết thúc trong khối try/finally (thành công hoặc bị ngắt).
        """
        if self.is_closed:
            return

        elapsed = time.time() - self.start_time
        hours, rem = divmod(elapsed, 3600)
        minutes, seconds = divmod(rem, 60)
        time_str = f"{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}"

        if self.status == "RUNNING":
            self.status = "COMPLETED"

        summary_lines = [
            "=" * 78,
            f">>> KẾT THÚC HUẤN LUYỆN: {self.config_name}",
            f"    Thời gian kết thúc:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"    Tổng thời gian chạy:  {elapsed:.2f}s ({time_str})",
            f"    Trạng thái cuối:      {self.status}",
            f"    Epoch hoàn tất cuối:  {self.last_completed_epoch}",
            f"    Checkpoint cuối:      {self.last_checkpoint_path or 'Chưa lưu'}",
        ]
        if final_metrics:
            summary_lines.append(f"    Kết quả đánh giá cuối:")
            for k, v in final_metrics.items():
                val_repr = f"{v:.4f}" if isinstance(v, float) else str(v)
                summary_lines.append(f"      - {k}: {val_repr}")
        summary_lines.append("=" * 78)

        for line in summary_lines:
            self.logger.info(line)

        summary_record = {
            "type": "summary",
            "config_name": self.config_name,
            "status": self.status,
            "total_duration_sec": round(elapsed, 2),
            "total_duration_formatted": time_str,
            "last_completed_epoch": self.last_completed_epoch,
            "last_checkpoint_path": self.last_checkpoint_path,
            "final_metrics": final_metrics,
            "end_timestamp": datetime.now().isoformat()
        }
        self._write_jsonl(summary_record)
        self.close()

    def register_signal_handler(self, model=None, checkpoint_dir=None, optimizer=None, scheduler=None):
        """
        Đăng ký xử lý tín hiệu SIGTERM (và SIGINT) để lưu checkpoint khẩn cấp
        (gồm weights mô hình, optimizer moments và scheduler state)
        và flush log ngay khi nhận tín hiệu sắp kill từ Queue Manager.
        """
        self.model_ref = model
        self.checkpoint_dir = os.path.normpath(os.path.abspath(checkpoint_dir)) if checkpoint_dir else None
        self.optimizer_ref = optimizer
        self.scheduler_ref = scheduler

        def _handle_signal(signum, frame):
            sig_name = "SIGTERM" if signum == signal.SIGTERM else ("SIGINT" if signum == signal.SIGINT else f"SIGNAL_{signum}")
            self.status = f"TERMINATED_{sig_name}"
            self.logger.warning(
                f"[CRITICAL] 🚨 Nhận tín hiệu {sig_name} từ hệ thống/Queue Manager! Đang lưu checkpoint khẩn cấp..."
            )
            checkpoint_epoch = self.last_completed_epoch
            self.logger.warning(
                f"Epoch đang chạy dở: {self.current_epoch} | Epoch hoàn tất cuối: {self.last_completed_epoch}"
            )
            self.logger.warning(
                f"💡 [RESUME GUIDE] Để chạy lại toàn bộ Epoch {self.current_epoch} bị gián đoạn, hãy chạy: "
                f"--resume-epoch {self.last_completed_epoch}"
            )

            emergency_path = None
            save_duration = None
            file_size_mb = None
            if self.model_ref is not None and self.checkpoint_dir is not None:
                try:
                    t_save_start = time.time()
                    os.makedirs(self.checkpoint_dir, exist_ok=True)
                    emergency_path = os.path.normpath(os.path.join(
                        self.checkpoint_dir,
                        f"emergency_checkpoint_epoch_{checkpoint_epoch}.pth"
                    ))
                    tmp_emergency_path = emergency_path + ".tmp"
                    import torch
                    if torch.cuda.is_available():
                        try:
                            torch.cuda.synchronize()
                        except Exception:
                            pass
                    checkpoint_data = {
                        "epoch": checkpoint_epoch,
                        "last_completed_epoch": self.last_completed_epoch,
                        "interrupted_epoch": self.current_epoch,
                        "model_state_dict": self.model_ref.state_dict(),
                    }
                    if self.optimizer_ref is not None:
                        try:
                            checkpoint_data["optimizer_state_dict"] = self.optimizer_ref.state_dict()
                        except Exception:
                            pass
                    if self.scheduler_ref is not None:
                        try:
                            checkpoint_data["scheduler_state_dict"] = self.scheduler_ref.state_dict()
                        except Exception:
                            pass
                    
                    # Ghi nguyên tử qua file tạm + flush + fsync + os.replace
                    with open(tmp_emergency_path, "wb") as f_chk:
                        torch.save(checkpoint_data, f_chk)
                        f_chk.flush()
                        os.fsync(f_chk.fileno())
                    os.replace(tmp_emergency_path, emergency_path)

                    save_duration = round(time.time() - t_save_start, 3)
                    file_size_mb = round(os.path.getsize(emergency_path) / (1024 * 1024), 2)
                    self.last_checkpoint_path = emergency_path
                    self.logger.warning(
                        f"[SAVED] ĐÃ LƯU CHECKPOINT KHẨN CẤP TẠI: {emergency_path} "
                        f"(Dung lượng: {file_size_mb} MB | Thời gian lưu: {save_duration}s)"
                    )
                except Exception as save_err:
                    self.logger.error(f"[ERROR] Không thể lưu checkpoint khẩn cấp: {save_err}")

            sig_record = {
                "type": "signal_termination",
                "signal": sig_name,
                "epoch_in_progress": self.current_epoch,
                "last_completed_epoch": self.last_completed_epoch,
                "emergency_checkpoint": emergency_path,
                "save_duration_sec": save_duration,
                "checkpoint_size_mb": file_size_mb,
                "timestamp": datetime.now().isoformat()
            }
            self._write_jsonl(sig_record)
            self.log_end()
            sys.exit(128 + signum)

        try:
            signal.signal(signal.SIGTERM, _handle_signal)
        except (ValueError, AttributeError):
            pass
        try:
            signal.signal(signal.SIGINT, _handle_signal)
        except (ValueError, AttributeError):
            pass

    def close(self):
        """Đóng các file handle một cách an toàn."""
        if self.is_closed:
            return
        self.is_closed = True

        try:
            if self.jsonl_file and not self.jsonl_file.closed:
                self.jsonl_file.flush()
                try:
                    os.fsync(self.jsonl_file.fileno())
                except Exception:
                    pass
                self.jsonl_file.close()
        except Exception:
            pass

        try:
            if self.file_handler:
                self.file_handler.flush()
                self.file_handler.close()
                self.logger.removeHandler(self.file_handler)
        except Exception:
            pass

        try:
            if self.stream_handler:
                self.stream_handler.flush()
                self.logger.removeHandler(self.stream_handler)
        except Exception:
            pass
