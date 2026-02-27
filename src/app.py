import cv2
import numpy as np
import time
import json

mode = 1
measure_psnr = False
recording = False
video_writer = None

fps_data = {1: [], 2: [], 3: []}
psnr_data = {1: [], 2: [], 3: []}

# ================= PSNR =================
def calculate_psnr(original, processed):
    mse = np.mean((original - processed) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))

# ================= Enhancement =================
def enhance_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 1.0)
    clahe = cv2.createCLAHE(2.0, (8, 8))
    return clahe.apply(gray)

# ================= Quantum Method =================
def quantum_edges(gray):

    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)

    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2

    u = np.arange(rows)
    v = np.arange(cols)
    U, V = np.meshgrid(u, v, indexing='ij')
    D = np.sqrt((U - crow)**2 + (V - ccol)**2)

    D0 = 20
    mask = 1 - np.exp(-(D**2)/(2*(D0**2)))
    fshift *= mask

    img_back = np.fft.ifft2(np.fft.ifftshift(fshift))
    img_back = np.abs(img_back)
    img_back = cv2.normalize(img_back,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)

    lap = cv2.Laplacian(img_back, cv2.CV_64F)
    lap = cv2.normalize(np.abs(lap),None,0,255,cv2.NORM_MINMAX).astype(np.uint8)

    _, edges = cv2.threshold(lap,50,255,cv2.THRESH_BINARY)
    return edges

# ================= Sobel =================
def sobel_edges(gray):
    gx=cv2.Sobel(gray,cv2.CV_64F,1,0)
    gy=cv2.Sobel(gray,cv2.CV_64F,0,1)
    mag=cv2.magnitude(gx,gy)
    return cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)

# ================= Main =================
def main():
    global mode, measure_psnr, recording, video_writer

    cap=cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera error")
        return

    print("""
Controls:
1 = Quantum
2 = Sobel
3 = Canny
p = Toggle PSNR
r = Start/Stop Recording
s = Save Frame
q = Quit
""")

    while True:

        start=time.time()

        ret,frame=cap.read()
        if not ret: break

        frame=cv2.resize(frame,(480,360))
        gray=enhance_image(frame)

        # algorithm select
        if mode==1:
            edges=quantum_edges(gray)
            name="Quantum"
        elif mode==2:
            edges=sobel_edges(gray)
            name="Sobel"
        else:
            edges=cv2.Canny(gray,100,200)
            name="Canny"

        # metrics
        end=time.time()
        fps=1/(end-start)
        latency=(end-start)*1000

        fps_data[mode].append(fps)

        if measure_psnr:
            psnr=calculate_psnr(gray,edges)
            psnr_data[mode].append(psnr)
        else:
            psnr=0

        combined=np.hstack((cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR),
                            cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR)))

        # display info
        cv2.putText(combined,f"Mode: {name}",(10,25),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

        cv2.putText(combined,f"FPS: {fps:.2f}",(10,55),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

        cv2.putText(combined,f"Latency: {latency:.2f} ms",(10,85),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

        if measure_psnr:
            cv2.putText(combined,f"PSNR: {psnr:.2f}",(10,115),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

        # recording
        if recording:
            video_writer.write(combined)

        cv2.imshow("Edge Detection System",combined)

        key=cv2.waitKey(1)&0xFF

        if key==ord('q'): break
        elif key==ord('1'): mode=1
        elif key==ord('2'): mode=2
        elif key==ord('3'): mode=3

        elif key==ord('p'):
            measure_psnr=not measure_psnr
            print("PSNR:",measure_psnr)

        elif key==ord('r'):
            recording=not recording
            if recording:
                video_writer=cv2.VideoWriter(
                    "recorded_output.avi",
                    cv2.VideoWriter_fourcc(*'XVID'),
                    20,
                    (combined.shape[1],combined.shape[0])
                )
                print("Recording started")
            else:
                video_writer.release()
                print("Recording stopped")

        elif key==ord('s'):
            cv2.imwrite("saved_frame.png",combined)
            print("Frame saved")

    cap.release()
    cv2.destroyAllWindows()

    with open("fps.json","w") as f:
        json.dump(fps_data,f)

    with open("psnr.json","w") as f:
        json.dump(psnr_data,f)

    print("Data saved.")

# ================= Run =================
if __name__=="__main__":
    main()
