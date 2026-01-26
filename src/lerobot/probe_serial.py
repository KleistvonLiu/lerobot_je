# 保存为 probe_serial.py 并运行：python3 probe_serial.py
import time, serial, binascii
PORT="/dev/ttyUSB0"
BAUDS=[115200,230400,460800,921600,1000000]
for b in BAUDS:
    try:
        print(f"\n=== {PORT}@{b} ===")
        ser=serial.Serial(PORT,b,timeout=0.1,rtscts=False,dsrdtr=False,xonxoff=False)
        # 尝试不同的 DTR/RTS 组合
        for dtr,rts in [(0,0),(1,1)]:
            ser.setDTR(bool(dtr)); ser.setRTS(bool(rts))
            time.sleep(0.1); ser.reset_input_buffer()
            ser.write(b'\n'); ser.flush()
            time.sleep(0.05)
            ser.write(b'U'*128); ser.flush()
            t0=time.time(); got=0; hdr=False
            while time.time()-t0<1.5:
                buf=ser.read(512)
                if buf:
                    got+=len(buf)
                    if b'\xff\x84' in buf: hdr=True
            print(f"DTR={dtr} RTS={rts} -> bytes={got}, ff84_header={hdr}")
        ser.close()
    except Exception as e:
        print("open failed:", e)