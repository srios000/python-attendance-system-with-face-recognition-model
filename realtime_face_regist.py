import cv2, os, tkinter as tk

faceCascade =cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

image_path = "img"
sample = "img/samples/"

#path declaration
if os.path.exists(image_path) == False:
    os.mkdir(image_path)
    
if os.path.exists(sample) == False:
    os.mkdir(sample)

名ボックス = tk.Tk()
名ボックス.geometry("300x100")
名ボックス.title("Name Input")
L1 = tk.Label(
    名ボックス, 
    text = "Enter your name here: ")
L1.pack()
E1 = tk.Entry(
    名ボックス, 
    width=20
)
E1.pack()

window2 = tk.Tk()
window2.geometry("275x175")
window2.resizable(width=False, height=False)
window2.title("Action")

# window.iconbitmap('assets/.ico')

def Search():
    cam = cv2.VideoCapture(0)
    while(cam.isOpened()): 
        _, frame = cam.read()
        face = faceCascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)
        for(x, y, w, h) in face:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.imwrite('img/samples/'+名前+'.jpg', frame)
            
        cv2.imshow("press 'enter' when you ready", frame)
        
        k = cv2.waitKey(1)
        if k == 13:
            break
            
    cam.release()
    cv2.destroyAllWindows()

# Button cari
button_ok = tk.Button(
    window2, 
    text="Register photo", 
    command=Search, 
    height=1, 
    width=34, 
    fg='#ffffff', 
    bg='#2da68c'
)

button_ok.grid(
    row=1, 
    padx=10, 
    pady=10
)

label_ok = tk.Label(
    window2, 
    text="one person per run", 
    relief = tk.RAISED, 
    font = ("Meiryo",8)
)

label_ok.grid(
    row=2, 
    padx=10, 
    pady=10
)

# Button Close
button_menu = tk.Button(
    window2, 
    text="Main Menu", 
    command=window2.quit, 
    height=1, 
    width=34, 
    fg='#ffffff', 
    bg='#2da68c'
)

button_menu.grid(
    row=3, 
    padx=10, 
    pady=10
)

def myClick1():
    global 名前
    名前 = str(E1.get())
    if len(E1.get()) == 0:
        window2.destroy()
        名ボックス.destroy()
    else:
        名ボックス.destroy()

button = tk.Button(
    名ボックス, 
    text="Confirm", 
    command=myClick1, 
    height=1, 
    width=15, 
    fg='#ffffff', 
    bg='#2da68c'
)

button.pack(side="bottom")


window2.mainloop()

名ボックス.mainloop()
