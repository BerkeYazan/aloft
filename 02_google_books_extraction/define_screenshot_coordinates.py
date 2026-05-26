import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import os
import json

# --- Configuration ---
# Two distinct sample images are needed to define coordinates for each category.
# Category 1: Filenames based on dates (e.g., '2025-xx-xx...')
SAMPLE_IMAGE_CAT1 = '2025-07-12_13.57.54.png'
# Category 2: Filenames based on book titles (e.g., '1Q84-1.png')
SAMPLE_IMAGE_CAT2 = '1Q84-1.png' 
CONFIG_FILE = 'coordinates.json'
MAX_DISPLAY_WIDTH = 2000
# ---

class CoordinatePicker:
    def __init__(self, master):
        self.master = master
        self.master.title("Coordinate Picker")
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.screenshots_dir = os.path.join(script_dir, 'Snippet_Screenshots')
        self.config_path = os.path.join(script_dir, CONFIG_FILE)

        self.canvas = tk.Canvas(master, cursor="cross")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)

        self.instructions = tk.Label(master, text="", font=("Helvetica", 14), bg="yellow", fg="black", padx=10, pady=5, relief="raised")
        self.instructions.pack(pady=10, fill="x")

        self.reset_state()
        self.process_next_category()

    def reset_state(self):
        self.clicks = []
        self.all_boxes = {}
        self.current_category_key = None
        self.current_box_name = None
        self.current_image_path = None
        self.category_queue = [
            ("date_based", SAMPLE_IMAGE_CAT1),
            ("title_based", SAMPLE_IMAGE_CAT2)
        ]

    def process_next_category(self):
        if not self.category_queue:
            self.save_and_show_results()
            return

        self.current_category_key, sample_image_name = self.category_queue.pop(0)
        self.current_image_path = os.path.join(self.screenshots_dir, sample_image_name)

        if not os.path.exists(self.current_image_path):
            messagebox.showerror("Error", f"Sample image not found: {self.current_image_path}")
            self.master.destroy()
            return

        self.all_boxes[self.current_category_key] = {}
        self.load_image_on_canvas()
        self.prompt_for_next_box()

    def load_image_on_canvas(self):
        self.original_image = Image.open(self.current_image_path)
        self.original_w, self.original_h = self.original_image.size
        self.scale_factor = MAX_DISPLAY_WIDTH / self.original_w
        display_w = MAX_DISPLAY_WIDTH
        display_h = int(self.original_h * self.scale_factor)

        self.display_image = self.original_image.resize((display_w, display_h), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(self.display_image)

        self.canvas.config(width=display_w, height=display_h)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

    def prompt_for_next_box(self):
        category_boxes = self.all_boxes[self.current_category_key]
        if "search_bar_box" not in category_boxes:
            self.current_box_name = "search_bar_box"
        elif "page_text_box" not in category_boxes:
            self.current_box_name = "page_text_box"
        else:
            self.process_next_category() # Move to the next image/category
            return
        
        self.instructions.config(text=f"For '{self.current_category_key}' images: Click the TOP-LEFT corner of the '{self.current_box_name}'.")

    def on_button_press(self, event):
        if self.current_box_name is None:
            return

        self.clicks.append((event.x, event.y))

        if len(self.clicks) == 1:
            self.instructions.config(text=f"Now click the BOTTOM-RIGHT corner of the '{self.current_box_name}'.")
        
        elif len(self.clicks) == 2:
            x1_disp, y1_disp = self.clicks[0]
            x2_disp, y2_disp = self.clicks[1]

            orig_x1 = int(x1_disp / self.scale_factor)
            orig_y1 = int(y1_disp / self.scale_factor)
            orig_x2 = int(x2_disp / self.scale_factor)
            orig_y2 = int(y2_disp / self.scale_factor)

            self.all_boxes[self.current_category_key][self.current_box_name] = (orig_x1, orig_y1, orig_x2, orig_y2)
            
            self.canvas.create_rectangle(x1_disp, y1_disp, x2_disp, y2_disp, outline='lime', width=3)
            
            self.clicks = []
            self.prompt_for_next_box()

    def save_and_show_results(self):
        self.current_box_name = None
        self.instructions.config(text="All done! You can now close this window.", bg="lightgreen", fg="black")
        
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.all_boxes, f, indent=4)
            result_string = f"Coordinates successfully saved to '{self.config_path}'!"
            print(result_string)
            print(json.dumps(self.all_boxes, indent=4))
            messagebox.showinfo("Success!", result_string)
        except Exception as e:
            messagebox.showerror("Error", f"Could not save coordinates file: {e}")

if __name__ == '__main__':
    root = tk.Tk()
    app = CoordinatePicker(root)
    root.mainloop() 