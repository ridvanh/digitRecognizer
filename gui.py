from keras.models import load_model
import customtkinter as ctk
from PIL import ImageGrab, Image
import numpy as np
import win32gui

model = load_model('mnist2.h5')
logo_img = Image.open("tdu_logo.png")

def predict_digit(img):
    img = img.resize((28, 28))
    img = img.convert('L')
    img = np.array(img)
    img = img.reshape(1, 28, 28, 1)
    img = (255 - img) / 255.0
    res = model.predict([img])[0]
    return np.argmax(res), max(res)


class App(ctk.CTk):
    def __init__(self):
        ctk.CTk.__init__(self)

        self.x = self.y = 0

        ctk.set_appearance_mode("light")

        self.title("Handschriftliche Ziffernerkennung")
        self.geometry('600x480')
        self.resizable(0, 0)

        self.canvas = ctk.CTkCanvas(self, width=300, height=300, bg="white", cursor="cross")
        self.label = ctk.CTkLabel(self, text="Warten", font=("MetaPro", 24))
        self.frame = ctk.CTkFrame(self, width=175, height=150, fg_color="transparent")
        self.classify_btn = ctk.CTkButton(self.frame, text="Eingeben", command=self.classify_handwriting)
        self.clear_button = ctk.CTkButton(self.frame, text="Löschen", command=self.clear_all)
        self.logo = ctk.CTkImage(logo_img, size=(120, 70))
        self.logo_label = ctk.CTkLabel(self, image=self.logo, text="")

        self.canvas.grid(row=1, column=3, pady=2, sticky=ctk.W, columnspan=1)
        self.label.grid(row=3, column=3, pady=10, padx=10)
        self.frame.grid(row=1, column=0)
        self.logo_label.grid(row=0, column=5, sticky='nw', padx=10, pady=10)

        self.classify_btn.pack(side="top", padx=10, pady=10)
        self.clear_button.pack(side="top", padx=10)

        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def clear_all(self):
        self.canvas.delete("all")

    def classify_handwriting(self):
        HWND = self.canvas.winfo_id()
        rect = win32gui.GetWindowRect(HWND)
        im = ImageGrab.grab(rect)
        digit, acc = predict_digit(im)
        self.label.configure(text=str(digit) + ', ' + str(int(acc * 100)) + '%')

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r = 8
        self.canvas.create_oval(self.x - r, self.y - r, self.x + r, self.y + r, fill='black')

app = App()
app.mainloop()
