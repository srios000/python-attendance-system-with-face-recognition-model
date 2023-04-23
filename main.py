from tkinter import *
import os

window = Tk()
window.geometry("605x350")
window.title("Attendance Program")

def FaceRegist():
    os.system("python realtime_face_regist.py")
def Attendance():
    os.system("python realtime_face_recognition.py")
def Emote():
    os.system("python realtime_face_emotion_detection.py")
def Help():
    os.system("start \"chrome\" \"readme.html\"")

# Label Judul
label_title = Label(window, text="Registration and Attendance System")
label_title.grid(row=0, column=0, columnspan=5, pady=20)

# Button Face Registration
button_d1 = Button(window, text="Face Registration", width=24, height=6, command=FaceRegist, fg='#ffffff', bg='#2da68c')
button_d1.grid(row=1, column=0, padx=10, pady=10)

# Button Attendance
button_d1 = Button(window, text="Attendance", width=24, height=6, command=Attendance, fg='#ffffff', bg='#2da68c')
button_d1.grid(row=1, column=1, padx=10, pady=10)

# Button Emotion Detection
button_d1 = Button(window, text="Emotion Detection", width=24, height=6, command=Emote, fg='#ffffff', bg='#2da68c')
button_d1.grid(row=1, column=2, padx=10, pady=10)

# Button Instruction
button_help = Button(window, text="Instruction (Read First)", command=Help, width=77, height=2, fg='#ffffff', bg='#2da68c')
button_help.grid(columnspan=4, padx=10, pady=10)

# Button Exit
button_quit = Button(window, text="Exit", command=window.quit, width=77, height=2, fg='#ffffff', bg='#2da68c')
button_quit.grid(columnspan=4, padx=10, pady=10)

window.mainloop()

