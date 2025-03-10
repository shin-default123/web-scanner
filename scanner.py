# Enhanced code

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import os
from fpdf import FPDF
import math


class DocumentScanner:
    def __init__(self, root):
        self.root = root
        self.root.title("Docs Scanner")
        self.processed_images = []
        self.current_index = -1
        self.capture_in_progress = False
        
        self.min_doc_area_percent = 10  
        self.doc_aspect_ratio_min = 0.5  
        self.doc_aspect_ratio_max = 2.0 
        
        
        self.create_widgets()

    def create_widgets(self):
        # Buttons frame
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)

        # Capture button
        self.capture_button = tk.Button(
            btn_frame, text="Capture Image", command=self.capture_image)
        self.capture_button.grid(row=0, column=0, padx=5)

        # Browse button
        self.browse_button = tk.Button(
            btn_frame, text="Browse Image", command=self.browse_image)
        self.browse_button.grid(row=0, column=1, padx=5)

        # Image display panel
        self.panel = tk.Label(self.root)
        self.panel.pack(pady=10)

        # Document detection status
        self.detection_status_label = tk.Label(self.root, text="Document not detected")
        self.detection_status_label.pack(pady=5)

        # Navigation frame
        self.nav_frame = tk.Frame(self.root)
        self.nav_frame.pack(pady=5)

        self.prev_button = tk.Button(
            self.nav_frame, text="Previous", command=self.show_previous_image, state=tk.DISABLED)
        self.prev_button.grid(row=0, column=0, padx=5)

        self.image_counter = tk.Label(self.nav_frame, text="No images")
        self.image_counter.grid(row=0, column=1, padx=5)

        self.next_button = tk.Button(
            self.nav_frame, text="Next", command=self.show_next_image, state=tk.DISABLED)
        self.next_button.grid(row=0, column=2, padx=5)

        # Save buttons frame
        self.save_frame = tk.Frame(self.root)
        self.save_frame.pack(pady=10)

        self.save_button = tk.Button(
            self.save_frame, text="Save as PDF", command=self.save_as_pdf, state=tk.DISABLED)
        self.save_button.grid(row=0, column=0, padx=5)

        self.save_jpg_button = tk.Button(
            self.save_frame, text="Save as JPG", command=self.save_as_jpg, state=tk.DISABLED)
        self.save_jpg_button.grid(row=0, column=1, padx=5)

        self.save_all_button = tk.Button(
            self.save_frame, text="Save All to PDF", command=self.save_all_as_pdf, state=tk.DISABLED)
        self.save_all_button.grid(row=0, column=2, padx=5)

        self.delete_button = tk.Button(
            self.save_frame, text="Delete Current", command=self.delete_current_image, state=tk.DISABLED)
        self.delete_button.grid(row=0, column=3, padx=5)
        
        # Add settings frame
        self.settings_frame = tk.LabelFrame(self.root, text="Scanner Settings")
        self.settings_frame.pack(pady=10, fill="x", padx=10)
        
        # Document detection sensitivity
        tk.Label(self.settings_frame, text="Min Document Size (%)").grid(row=0, column=0, padx=5, pady=5)
        self.min_area_slider = tk.Scale(self.settings_frame, from_=1, to=50, orient=tk.HORIZONTAL)
        self.min_area_slider.set(self.min_doc_area_percent)  
        self.min_area_slider.grid(row=0, column=1, padx=5, pady=5)
        
        # Add enhancement settings
        self.enhance_var = tk.BooleanVar(value=True)
        self.enhance_check = tk.Checkbutton(self.settings_frame, text="Enhance Document", 
                                          variable=self.enhance_var)
        self.enhance_check.grid(row=1, column=0, columnspan=2, padx=5, pady=5)

    def order_points(self, pts):
        """Order points in clockwise order starting from top-left"""
        rect = np.zeros((4, 2), dtype=np.float32)
        
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        return rect

    def capture_image(self):
        self.capture_in_progress = True
        cap = cv2.VideoCapture(0)
        
        ret, test_frame = cap.read()
        if ret:
            frame_height, frame_width = test_frame.shape[:2]
            total_frame_area = frame_width * frame_height
        else:
            self.capture_in_progress = False
            return

        doc_corners = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if self.capture_in_progress:
                display_frame = frame.copy()
                
                self.min_doc_area_percent = self.min_area_slider.get()
                min_doc_area = (self.min_doc_area_percent / 100) * total_frame_area
                
                doc_corners = self.detect_document(frame, min_doc_area)
                
                if doc_corners is not None:
                    cv2.drawContours(display_frame, [doc_corners.astype(np.int32)], -1, (0, 255, 0), 2)
                    
                    for point in doc_corners:
                        x, y = point.astype(int)
                        cv2.circle(display_frame, (x, y), 5, (0, 0, 255), -1)
                        
                    self.detection_status_label.config(text="Document Detected!", fg="green")
                else:
                    self.detection_status_label.config(text="No Document Detected", fg="red")

                cv2.imshow('Press Space to Capture, ESC to Exit', display_frame)

            key = cv2.waitKey(1)

            if key & 0xFF == ord(' '):  # Space key to capture
                if doc_corners is not None:
                    warped = self.four_point_transform(frame, doc_corners)
                    self.capture_in_progress = False
                    break
                else:
                    border_frame = display_frame.copy()
                    cv2.rectangle(border_frame, (0, 0), (frame_width, frame_height), (0, 0, 255), 20)
                    cv2.imshow('Press Space to Capture, ESC to Exit', border_frame)
                    cv2.waitKey(100)
                    
            elif key & 0xFF == 27:  # ESC key to exit
                self.capture_in_progress = False
                cap.release()
                cv2.destroyAllWindows()
                return

        cap.release()
        cv2.destroyAllWindows()

        if doc_corners is not None:
            temp_filename = f"temp_capture_{len(self.processed_images)}.jpg"
            cv2.imwrite(temp_filename, warped)
            
            self.process_captured_image(temp_filename)
            os.remove(temp_filename)

    def detect_document(self, frame, min_area):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)
        
        edges1 = cv2.Canny(blurred, 50, 200, apertureSize=3)
        
        _, thresh1 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        edges = cv2.bitwise_or(edges1, thresh1)
        
        kernel = np.ones((3, 3), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)
        
        contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            morphed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, morph_kernel)
            edges2 = cv2.Canny(morphed, 30, 100)
            contours, _ = cv2.findContours(edges2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                  cv2.THRESH_BINARY_INV, 11, 2)
            contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
        if not contours:
            return None
            
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        if not valid_contours:
            return None
            
        best_doc = None
        max_score = -1
        
        for cnt in sorted(valid_contours, key=cv2.contourArea, reverse=True)[:5]:  # Check top 5 contours
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            
            if len(approx) == 4:
                area = cv2.contourArea(approx)
                
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h if h != 0 else 0
                
                if self.doc_aspect_ratio_min <= aspect_ratio <= self.doc_aspect_ratio_max:
                    hull = cv2.convexHull(approx)
                    hull_area = cv2.contourArea(hull)
                    convexity = area / hull_area if hull_area > 0 else 0
                    
                    rect_area = w * h
                    rectangularity = area / rect_area if rect_area > 0 else 0
                    
                    score = area * convexity * rectangularity
                    
                    if score > max_score:
                        max_score = score
                        best_doc = approx
        
        if best_doc is not None:
            best_doc = best_doc.reshape(4, 2).astype(np.float32)
            return self.order_points(best_doc)
            
        return None

    def four_point_transform(self, image, pts):
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect
        
        # Width of new image
        width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        max_width = max(int(width_a), int(width_b))
        
        # Height of new image
        height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_height = max(int(height_a), int(height_b))
        
        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype=np.float32)
        
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (max_width, max_height))
        
        return warped
    
    # Browse images
    def browse_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff")])
        if file_path:
            self.process_image(file_path)

    def process_captured_image(self, image_path):
        image = cv2.imread(image_path)
        
        if self.enhance_var.get():
            enhanced = self.enhance_document(image)
        else:
            enhanced = image
            
        out_filename = f"processed_image_{len(self.processed_images)}.jpg"
        cv2.imwrite(out_filename, enhanced)
        self.processed_images.append(out_filename)

        self.current_index = len(self.processed_images) - 1
        self.show_current_image()
        self.update_navigation()

    def process_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return
            
        height, width = image.shape[:2]
        min_doc_area = (self.min_doc_area_percent / 100) * (height * width)
        
        doc_corners = self.detect_document(image, min_doc_area)
        
        if doc_corners is not None:
            warped = self.four_point_transform(image, doc_corners)
        else:
            print("No document detected in the image. Using original image.")
            warped = image
            
        # Enhance image if enabled
        if self.enhance_var.get():
            enhanced = self.enhance_document(warped)
        else:
            enhanced = warped
            
        out_filename = f"processed_image_{len(self.processed_images)}.jpg"
        cv2.imwrite(out_filename, enhanced)
        self.processed_images.append(out_filename)

        self.current_index = len(self.processed_images) - 1
        self.show_current_image()
        self.update_navigation()

    def enhance_document(self, image):
        """Enhance document image quality"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(gray)
        
        denoised = cv2.bilateralFilter(equalized, 9, 75, 75)
        
        thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        if len(image.shape) == 3:
            result = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        else:
            result = thresh
            
        return result

    def show_current_image(self):
        if self.current_index >= 0 and self.current_index < len(self.processed_images):
            img = Image.open(self.processed_images[self.current_index])
            img = img.resize((400, 500))
            img_tk = ImageTk.PhotoImage(img)
            self.panel.config(image=img_tk)
            self.panel.image = img_tk
            self.image_counter.config(
                text=f"Image {self.current_index + 1} of {len(self.processed_images)}")
            self.enable_save_buttons()
        else:
            self.panel.config(image='')
            self.image_counter.config(text="No images")
            self.disable_save_buttons()

    def show_next_image(self):
        if self.current_index < len(self.processed_images) - 1:
            self.current_index += 1
            self.show_current_image()
            self.update_navigation()

    def show_previous_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.show_current_image()
            self.update_navigation()

    def update_navigation(self):
        if self.current_index <= 0:
            self.prev_button.config(state=tk.DISABLED)
        else:
            self.prev_button.config(state=tk.NORMAL)

        if self.current_index >= len(self.processed_images) - 1:
            self.next_button.config(state=tk.DISABLED)
        else:
            self.next_button.config(state=tk.NORMAL)

        if len(self.processed_images) > 1:
            self.save_all_button.config(state=tk.NORMAL)
        else:
            self.save_all_button.config(state=tk.DISABLED)

        if len(self.processed_images) > 0:
            self.delete_button.config(state=tk.NORMAL)
        else:
            self.delete_button.config(state=tk.DISABLED)

    def enable_save_buttons(self):
        self.save_button.config(state=tk.NORMAL)
        self.save_jpg_button.config(state=tk.NORMAL)

    def disable_save_buttons(self):
        self.save_button.config(state=tk.DISABLED)
        self.save_jpg_button.config(state=tk.DISABLED)
        self.save_all_button.config(state=tk.DISABLED)
        self.delete_button.config(state=tk.DISABLED)

    def save_as_pdf(self):
        if self.current_index >= 0:
            pdf = FPDF()
            pdf.add_page()

            img = Image.open(self.processed_images[self.current_index])
            img = img.resize((int(210 * 3.8), int(297 * 3.8)))

            temp_path = "temp_for_pdf.jpg"
            img.save(temp_path)

            pdf.image(temp_path, 0, 0, 210, 297)

            pdf_file = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")])
            if pdf_file:
                pdf.output(pdf_file)

            os.remove(temp_path)

    def save_all_as_pdf(self):
        if not self.processed_images:
            return

        pdf = FPDF()

        pdf_file = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")])
        if not pdf_file:
            return

        temp_dir = "temp_pdf_images"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        try:
            for i, img_path in enumerate(self.processed_images):
                pdf.add_page()

                img = Image.open(img_path)
                img = img.resize((int(210 * 3.8), int(297 * 3.8)))

                temp_path = os.path.join(temp_dir, f"temp_pdf_image_{i}.jpg")
                img.save(temp_path)

                pdf.image(temp_path, 0, 0, 210, 297) 

            # Save the final PDF
            pdf.output(pdf_file)
        
        finally:
            # Clean up all temporary files
            for i in range(len(self.processed_images)):
                temp_path = os.path.join(temp_dir, f"temp_pdf_image_{i}.jpg")
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)

    def save_as_jpg(self):
        if self.current_index >= 0:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".jpg", filetypes=[("JPEG files", "*.jpg")])
            if file_path:
                img = Image.open(self.processed_images[self.current_index])
                img.save(file_path)

    def delete_current_image(self):
        if self.current_index >= 0 and len(self.processed_images) > 0:
            os.remove(self.processed_images[self.current_index])
            del self.processed_images[self.current_index]

            # Update current index
            if len(self.processed_images) == 0:
                self.current_index = -1
            elif self.current_index >= len(self.processed_images):
                self.current_index = len(self.processed_images) - 1

            self.show_current_image()
            self.update_navigation()

    def cleanup(self):
        for img_path in self.processed_images:
            if os.path.exists(img_path):
                os.remove(img_path)


if __name__ == "__main__":
    root = tk.Tk()
    app = DocumentScanner(root)

    def on_closing():
        app.cleanup()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()