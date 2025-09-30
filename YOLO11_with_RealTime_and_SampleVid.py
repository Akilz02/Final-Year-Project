import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
import threading
import time
import serial

class YOLOApp:
    def __init__(self, root, model_path):
        self.root = root
        self.root.title("🔍 YOLO Real-Time Detection")
        self.root.geometry("1920x1080")
        self.root.configure(bg="#282c34")
        self.video_running = False
        self.cap = None
        self.current_imgtk = None

        # Load YOLO model
        self.model = YOLO(model_path)

        # Open Serial to ESP32
        try:
            self.esp = serial.Serial('COM3', 115200, timeout=1)
            print("✅ ESP32 connected via Serial.")
        except Exception as e:
            self.esp = None
            print("⚠️ ESP32 not connected:", e)

        # UI
        self.title_frame = tk.Frame(self.root, bg="#282c34", pady=15)
        self.title_frame.pack(fill=tk.X, side=tk.TOP)

        self.title_label = tk.Label(
            self.title_frame,
            text="🧠 YOLO Real-Time Object Detection",
            font=("Segoe UI", 24, "bold"),
            bg="#282c34",
            fg="#61dafb"
        )
        self.title_label.pack()

        self.status_label = tk.Label(
            self.root,
            text="Status: Idle",
            font=("Segoe UI", 13),
            bg="#282c34",
            fg="#ABB2BF",
            pady=10
        )
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

        self.btn_frame = tk.Frame(self.root, bg="#282c34", pady=15)
        self.btn_frame.pack(side=tk.BOTTOM, pady=10)

        button_font = ("Segoe UI", 14, "bold")

        self.start_btn = tk.Button(
            self.btn_frame,
            text="▶ Start Detection (Webcam)",
            command=self.start_detection,
            bg="#4CAF50",
            fg="white",
            font=button_font,
            width=22
        )
        self.start_btn.grid(row=0, column=0, padx=25)

        self.sample_btn = tk.Button(
            self.btn_frame,
            text="🎥 Play Sample Video",
            command=self.start_sample_video,
            bg="#2196F3",
            fg="white",
            font=button_font,
            width=22
        )
        self.sample_btn.grid(row=0, column=1, padx=25)

        self.stop_btn = tk.Button(
            self.btn_frame,
            text="⏹ Stop Detection",
            command=self.stop_detection,
            bg="#F44336",
            fg="white",
            font=button_font,
            width=22,
            state=tk.DISABLED
        )
        self.stop_btn.grid(row=0, column=2, padx=25)

        self.label = tk.Label(self.root, bg="#3b4048", bd=2, relief="solid")
        self.label.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)

    def start_detection(self):
        """ Start webcam detection """
        if self.video_running:
            return

        self.cap = cv2.VideoCapture(1)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open webcam.")
                self.stop_detection()
                return

        # Webcam resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        self.video_running = True
        self.start_btn.config(state=tk.DISABLED)
        self.sample_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Status: Detecting (Webcam)...", fg="#61dafb")

        threading.Thread(target=self.video_loop, daemon=True).start()

    def start_sample_video(self):
        """ Start detection on sample video file """
        if self.video_running:
            return

        video_path = r"C:\Users\ASUS\Desktop\APIIT\FYP\sample_video.mp4"
        # video_path = r"C:\Users\ASUS\Desktop\APIIT\FYP\pothole_sample02.mp4"
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open sample video.")
            return

        self.video_running = True
        self.start_btn.config(state=tk.DISABLED)
        self.sample_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Status: Playing Sample Video...", fg="#61dafb")

        threading.Thread(target=self.video_loop, daemon=True).start()

    def stop_detection(self):
        """ Stop video or webcam detection """
        if not self.video_running:
            return

        self.video_running = False
        time.sleep(0.1)

        if self.cap:
            self.cap.release()
            self.cap = None

        self.label.config(image='')
        self.current_imgtk = None
        self.start_btn.config(state=tk.NORMAL)
        self.sample_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Status: Idle", fg="#ABB2BF")

    def video_loop(self):
        """ Video processing loop """
        while self.video_running:
            ret, frame = self.cap.read()
            if not ret:
                self.root.after(0, self.stop_detection)
                break

            results = self.model(frame)

            led_indices = []
            frame_width = frame.shape[1]

            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id].lower()

                if "pothole" in label:
                    x_center = int((box.xyxy[0][0] + box.xyxy[0][2]) / 2)
                    led_index = int((x_center / frame_width) * 144)
                    led_index = max(0, min(143, led_index))
                    led_indices.append(led_index)

            # Send LED indices to ESP32
            if self.esp:
                try:
                    if led_indices:
                        msg = ",".join(str(i) for i in led_indices)
                    else:
                        msg = ""  # clear all LEDs
                    self.esp.write((msg + "\n").encode())
                    print(f"Sent to ESP32: {msg if msg else '[CLEAR]'}")
                    time.sleep(0.05)
                except Exception as e:
                    print("Error sending to ESP32:", e)

            # Display annotated frame
            annotated_frame = results[0].plot()

            label_width = self.label.winfo_width()
            label_height = self.label.winfo_height()

            if label_width <= 1 or label_height <= 1:
                display_width = 900
                display_height = 550
            else:
                original_h, original_w, _ = annotated_frame.shape
                aspect_ratio = original_w / original_h

                if (label_width / aspect_ratio) <= label_height:
                    display_width = label_width
                    display_height = int(label_width / aspect_ratio)
                else:
                    display_height = label_height
                    display_width = int(label_height * aspect_ratio)

                display_width = max(1, display_width)
                display_height = max(1, display_height)

            display_frame = cv2.resize(annotated_frame, (display_width, display_height))
            rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(image=img)

            self.root.after(0, self.update_display, imgtk)

        if self.cap:
            self.cap.release()
            self.cap = None

    def update_display(self, imgtk):
        if self.video_running:
            self.current_imgtk = imgtk
            self.label.configure(image=imgtk)


if __name__ == "__main__":
    model_path = r"C:\Users\ASUS\Desktop\APIIT\FYP\YOLOV11\model_- 29 may 2025 10_31.pt"
    root = tk.Tk()
    app = YOLOApp(root, model_path)
    root.mainloop()
