#!/usr/bin/env python3
# fast_visualize_tactile_cv2_with_right_label.py
import time, sys, argparse, struct, os
import numpy as np
import serial
import cv2

# --- 可选：用 Pillow 画中文 ---
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_OK = True
except Exception:
    PIL_OK = False

HEADER     = b'\xFF\x84'
FRAME_SIZE = 70
ROWS, COLS = 4, 8
CH         = ROWS * COLS

MIN_VAL = 0.0
MAX_VAL = 100.0

# -------- 串口帧读取（缓冲找帧头，低开销） ----------
class FrameReader:
    def __init__(self, port, baud, read_chunk=1024, timeout=0.02):
        self.ser = serial.Serial(
            port=port, baudrate=baud,
            bytesize=8, parity='N', stopbits=1,
            timeout=timeout, write_timeout=0.2,
            rtscts=False, dsrdtr=False, xonxoff=False
        )
        try:
            self.ser.dtr = True
            self.ser.rts = False
        except Exception:
            pass
        self.buf = bytearray()
        self.read_chunk = read_chunk
        self.ser.reset_input_buffer()

    def close(self):
        try: self.ser.close()
        except Exception: pass

    @staticmethod
    def _verify_checksum(frame: memoryview) -> bool:
        s = 0
        for b in frame[2:68]:  # 2..67 求和
            s += b
        calc = s & 0xFFFF
        recv = (frame[68] << 8) | frame[69]
        return calc == recv

    def read_frame(self, deadline_s=0.5):
        end = time.monotonic() + deadline_s
        while time.monotonic() < end:
            chunk = self.ser.read(self.read_chunk)
            if chunk:
                self.buf.extend(chunk)
            else:
                time.sleep(0.001)

            start = self.buf.find(HEADER)
            while start != -1:
                avail = len(self.buf) - start
                if avail < FRAME_SIZE:
                    break
                frame = self.buf[start:start+FRAME_SIZE]
                del self.buf[:start+FRAME_SIZE]
                mv = memoryview(frame)
                if self._verify_checksum(mv):
                    return bytes(frame)
                start = self.buf.find(HEADER)
        return None

_unpack_fmt = '>' + 'H'*CH

def parse_adc(frame: bytes):
    cnt = struct.unpack_from('>H', frame, 2)[0]
    adc = struct.unpack_from(_unpack_fmt, frame, 4)
    return cnt, np.frombuffer(np.asarray(adc, dtype=np.uint16), dtype=np.uint16).astype(np.float32, copy=False)

def build_index_map(reverse_rows=True, flipud=False, fliplr=False):
    grid = np.arange(ROWS*COLS).reshape(ROWS, COLS)
    if reverse_rows: grid = grid[::-1, :]
    if flipud:       grid = np.flipud(grid)
    if fliplr:       grid = np.fliplr(grid)
    return grid.ravel()

# -------- OpenCV 可视化（右侧中文竖排面板） ----------
class VisualizerCV2:
    def __init__(self, title="Tactile", scale=80, mode="heatmap", show="mapped",
                 font=None, decimals=0, draw_grid=True,
                 right_text="接口端", right_width=None, font_path=None):
        """
        scale: 每格像素（例如 80 -> 每触点 80x80 像素）
        mode:  'heatmap' 或 'text'
        show:  'mapped' or 'raw'（仅 text 模式有意义）
        right_text: 右侧竖排中文（默认“接口端”）
        right_width: 右侧面板宽度（像素）。None 则取 int(scale*0.9)
        font_path: 指定中文字体路径（如 NotoSansCJK / 思源黑体）；若不提供将自动猜测
        """
        self.title     = title
        self.scale     = int(scale)
        self.mode      = mode
        self.show      = show
        # OpenCV 自带字体（给 text 模式数字用，不影响中文）
        if font is None:
            self.font = getattr(cv2, "FONT_HERSHEY_SIMPLEX",
                         getattr(cv2, "FONT_HERSHEY_PLAIN", 0))
        else:
            self.font = font
        self.decimals  = decimals
        self.draw_grid = draw_grid

        self.H = ROWS * self.scale
        self.W = COLS * self.scale

        # 右侧中文面板（预渲染一次，后续直接拼接）
        self.right_text = right_text
        self.right_w    = int(right_width if right_width is not None else max(40, self.scale * 9 // 10))
        self.right_panel_bgr = self._build_right_panel(self.H, self.right_w, right_text, font_path)

        self.W_total = self.W + self.right_w

        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(self.title, self.W_total, self.H)

        # 预计算每格中心（text 模式数字用）
        self.text_pos = []
        for r in range(ROWS):
            for c in range(COLS):
                x = int((c + 0.5) * self.scale)
                y = int((r + 0.6) * self.scale)
                self.text_pos.append((x, y))

    def _pick_font_path(self, user_font: str | None):
        if user_font and os.path.exists(user_font):
            return user_font
        # 常见中文字体候选（Ubuntu / Debian）
        candidates = [
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/truetype/arphic/ukai.ttc",
            "/usr/share/fonts/truetype/simhei.ttf",
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        return None  # 找不到就让 Pillow 用默认（可能显示为方块）

    def _build_right_panel(self, height, width, text_cn: str, font_path: str | None):
        # 背景淡灰
        panel = np.full((height, width, 3), 235, dtype=np.uint8)

        if not PIL_OK:
            # 没有 Pillow：退化为竖排英文占位
            fallback = "J\nI\nE\nK\nO\nU\nD\nU\nA\nN"
            y_step = height // (len(fallback.split("\n")) + 1)
            y = y_step
            for ch in fallback.split("\n"):
                (w, h), _ = cv2.getTextSize(ch, self.font, 0.6, 1)
                x = width // 2 - w // 2
                cv2.putText(panel, ch, (x, y), self.font, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
                y += y_step
            return panel

        # 有 Pillow：用中文字体竖排绘制
        fp = self._pick_font_path(font_path)
        try:
            # 字号：跟随 scale，留出上下边距
            size = max(14, int(self.scale * 0.6))
            font = ImageFont.truetype(fp, size=size) if fp else ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()

        img_pil = Image.fromarray(panel[..., ::-1])  # BGR->RGB
        draw = ImageDraw.Draw(img_pil)

        # 竖排：均匀分布在高度方向
        chars = list(text_cn) if text_cn else list("接口端")
        n = len(chars)
        y_step = height / (n + 1)
        for i, ch in enumerate(chars, start=1):
            # 计算文字尺寸以水平居中
            try:
                bbox = draw.textbbox((0, 0), ch, font=font)
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
            except Exception:
                w, h = draw.textlength(ch, font=font), size
            x = int((width - w) / 2)
            y = int(i * y_step - h / 2)
            draw.text((x, y), ch, fill=(0, 0, 0), font=font)

        panel_rgb = np.asarray(img_pil)
        return panel_rgb[..., ::-1]  # RGB->BGR

    def render_heatmap(self, vals_0_100):
        img = (255.0 - np.clip(vals_0_100, 0, 100) * 2.55).astype(np.uint8).reshape(ROWS, COLS)
        img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if self.draw_grid:
            for r in range(1, ROWS):
                y = r * self.scale
                cv2.line(img, (0, y), (self.W, y), (80, 80, 80), 1, cv2.LINE_AA)
            for c in range(1, COLS):
                x = c * self.scale
                cv2.line(img, (x, 0), (x, self.H), (80, 80, 80), 1, cv2.LINE_AA)

        # 拼上右侧中文面板
        out = np.hstack([img, self.right_panel_bgr])
        cv2.imshow(self.title, out)

    def render_text(self, vals_0_100, raw_adc=None):
        img = np.full((self.H, self.W, 3), 235, dtype=np.uint8)

        if self.draw_grid:
            for r in range(1, ROWS):
                y = r * self.scale
                cv2.line(img, (0, y), (self.W, y), (160, 160, 160), 1, cv2.LINE_AA)
            for c in range(1, COLS):
                x = c * self.scale
                cv2.line(img, (x, 0), (x, self.H), (160, 160, 160), 1, cv2.LINE_AA)

        if self.show == "raw" and raw_adc is not None:
            texts = [str(int(v)) for v in raw_adc]
        else:
            fmt = "{:." + str(self.decimals) + "f}"
            texts = [fmt.format(float(v)) for v in vals_0_100]

        for (x, y), s in zip(self.text_pos, texts):
            cv2.putText(img, s, (x, y), self.font, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

        out = np.hstack([img, self.right_panel_bgr])
        cv2.imshow(self.title, out)

def main():
    ag = argparse.ArgumentParser("Ultra-fast tactile visualizer (OpenCV backend) + right Chinese label")
    ag.add_argument("--port", default="/dev/serial/by-id/usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0")
    ag.add_argument("--baud", type=int, default=460800)
    ag.add_argument("--timeout", type=float, default=0.5)

    ag.add_argument("--mode", choices=["heatmap", "text"], default="heatmap")
    ag.add_argument("--display", choices=["mapped", "raw"], default="mapped")  # text 模式有效
    ag.add_argument("--adc-min", type=float, default=None)
    ag.add_argument("--adc-max", type=float, default=None)

    ag.add_argument("--flipud", action="store_true")
    ag.add_argument("--fliplr", action="store_true")
    ag.add_argument("--scale", type=int, default=80)
    ag.add_argument("--decimals", type=int, default=0)

    # 右侧中文参数
    ag.add_argument("--right-text", default="接口端")
    ag.add_argument("--right-width", type=int, default=None, help="右侧面板像素宽度，默认 ~0.9*scale")
    ag.add_argument("--font", dest="font_path", default=None, help="中文字体文件路径（如 NotoSansCJK / 思源黑体）")
    args = ag.parse_args()

    # 预计算索引映射（默认把 4 行翻转）
    idx_map = build_index_map(reverse_rows=True, flipud=args.flipud, fliplr=args.fliplr)

    # 线性映射 0..100
    use_map = args.adc_min is not None and args.adc_max is not None and args.adc_max > args.adc_min
    if use_map:
        scale = 100.0 / (args.adc_max - args.adc_min)
        bias  = -args.adc_min * scale
    else:
        scale = 1.0
        bias  = 0.0

    try:
        fr = FrameReader(args.port, args.baud)
    except serial.SerialException as e:
        print(f"[串口异常] {e}", file=sys.stderr); sys.exit(1)

    vis = VisualizerCV2(title="Tactile", scale=args.scale,
                        mode=args.mode, show=args.display, decimals=args.decimals,
                        right_text=args.right_text, right_width=args.right_width, font_path=args.font_path)

    target_dt = 1.0 / 20.0
    last_show = 0.0

    try:
        while True:
            frame = fr.read_frame(deadline_s=args.timeout)
            if frame is None:
                if cv2.waitKey(1) == 27:  # ESC 退出
                    break
                continue

            cnt, adc = parse_adc(frame)
            vals = adc * scale + bias if use_map else adc
            # np.clip(vals, MIN_VAL, MAX_VAL, out=vals)  # 如需严格 0..100 可开启

            ordered = np.take(vals, idx_map, mode='clip')  # 4x8 顺序
            if time.monotonic() - last_show >= target_dt:
                if args.mode == "heatmap":
                    vis.render_heatmap(ordered)
                else:
                    if args.display == "raw":
                        ordered_raw = np.take(adc, idx_map, mode='clip')
                        vis.render_text(ordered, raw_adc=ordered_raw)
                    else:
                        vis.render_text(ordered, raw_adc=None)
                last_show = time.monotonic()

            k = cv2.waitKey(1)
            if k in (27, ord('q'), ord('Q')):
                break
    except KeyboardInterrupt:
        pass
    finally:
        fr.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
