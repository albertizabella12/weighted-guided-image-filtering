import cv2
import numpy as np
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk


#CALCULUL PONDERII SENSIBILE LA MARGINI = calculeaza harta de ponderi (gamma)
#masoara local cat de neteda sau detaliata este o parte din imaginea de ghid


def compute_edge_aware_weighting(G, r_small=1, L=255, s=1.0):
    if G.dtype != np.float64:
        G = G.astype(np.float64)
        #asigurare ca imaginea de ghid este de tip float(pt precizie)


    #dimensiunea kernel-ului (ferestrei) pentru calculul variantei locale
    ksize_small = (2 * r_small + 1, 2 * r_small + 1)

    epsilon = (0.001 * L) ** 2
    #N=nr total de pixeli din imagine (pt. normalizare)
    N = G.size


    #calcul medii locale
    mean_G = cv2.boxFilter(G, cv2.CV_64F, ksize_small, normalize=True)
    mean_G2 = cv2.boxFilter(G * G, cv2.CV_64F, ksize_small, normalize=True)

    #varianta locala
    var_G_small = mean_G2 - mean_G * mean_G
    numerator = 1.0 / (var_G_small + epsilon)
    suma = np.sum(numerator) / N

    #calculeaza harta de ponderi gamma dupa formula 5
    gamma = (var_G_small + epsilon) * suma
    #se realizeaza o netezire Gaussiana
    #s este deviatia standard (Smooth) controlata de sliderul "Gamma"
    gamma_smoothed = cv2.GaussianBlur(gamma, (0, 0), s)

    return gamma_smoothed

#functie care implementeaza calcului stratului de baza Z
def weighted_guided_filter(X, G, gamma_map, r, lambda_reg):
    # X: imaginea de intrare
    # G: imaginea de ghid
    # r: raza ferestrei
    # lambda_reg = regularizarea

    if X.dtype != np.float64:
        X = X.astype(np.float64)
    if G.dtype != np.float64:
        G = G.astype(np.float64)

    #raza ferestrei, ex:r=1 vom avea o matrice (3,3)
    ksize = (2 * r + 1, 2 * r + 1)

    #calcul medii locale
    #boxFilter = medie aritmetica intre toti pixelii
    mean_X = cv2.boxFilter(X, cv2.CV_64F, ksize, normalize=True)
    mean_G = cv2.boxFilter(G, cv2.CV_64F, ksize, normalize=True)
    mean_GX = cv2.boxFilter(G * X, cv2.CV_64F, ksize, normalize=True)
    mean_GG = cv2.boxFilter(G * G, cv2.CV_64F, ksize, normalize=True)

    #varianta locala a ghidului
    #cat se schimba pixelii ghidului intr-o zona
    var_G = mean_GG - mean_G * mean_G

    #covarianta intre ghid si canal
    #cat de mult se schimba X si G
    cov_GX = mean_GX - mean_G * mean_X


    #a = cat de mult ramane din margini
    a = cov_GX / (var_G + (lambda_reg / gamma_map))
    b = mean_X - a * mean_G

    #netezire a si b => versiuni mai line ale coeficientilor
    mean_a = cv2.boxFilter(a, cv2.CV_64F, ksize, normalize=True)
    mean_b = cv2.boxFilter(b, cv2.CV_64F, ksize, normalize=True)

    Z_hat = mean_a * G + mean_b
    #formula finala
    return Z_hat



# GUI SETUP
#culorile folosite
COLORS = {
    "bg_main": "#2b2b2b",
    "bg_panel": "#383838",
    "text": "#ffffff",
    "accent_1": "#3498db",
    "accent_2": "#27ae60",
    "slider_trough": "#505050"
}

root = Tk()
root.title("WGIF") #titlul
root.geometry("1300x750")  #dimensiunea
root.configure(bg=COLORS["bg_main"])  #culoare fundal
# Facem fereastra minimă ca să nu se strice layout-ul dacă o micșorezi prea tare
root.minsize(1000, 600)

input_image = None
output_image = None
#variabile globale pt imaginea incarcata si imaginea filtrata


#APLICpai ATIE
#deschide alegerea unui fisier
def open_image():
    global input_image
    file_path = filedialog.askopenfilename()

    if not file_path:
        return

    #daca nu s-a ales nimic, se iese din functie


    #citeste imaginea
    input_image = cv2.imread(file_path)

    # Afișăm imaginea in panoul original
    show_image(input_image, panel_original_lbl)

    # Resetăm rezultatele vizual
    clear_panel(panel_base_lbl, "Base Layer")
    clear_panel(panel_enhanced_lbl, "Enhanced")


#functia de procesare a imaginii
def enhance_image():
    global input_image, output_image

    #daca nu s-a incarcat nicio imagine => iesirea din functie
    if input_image is None:
        return

    # Valori slidere
    r_large = int(slider_r.get())    #raza ferestrei pentru Guided Filter
    theta = float(slider_theta.get()) #amplificarea detaliilor
    s_gamma = float(slider_gamma.get())  #smoothness pt gaussian blur

    # Procesare WGIF, convertire in grayscale
    img_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    #calculam harta gamma cu ajutorul functiei
    gamma_map = compute_edge_aware_weighting(img_gray, r_small=1, L=255, s=s_gamma)


    #impartim imaginea in canale B G R
    B, G, R = cv2.split(input_image)
    lambda_reg = 1 / 256


    #calculul stratului de baza pt fiecare canal cu ajutorul functiei
    B_base = weighted_guided_filter(B, img_gray, gamma_map, r_large, lambda_reg)
    G_base = weighted_guided_filter(G, img_gray, gamma_map, r_large, lambda_reg)
    R_base = weighted_guided_filter(R, img_gray, gamma_map, r_large, lambda_reg)


    #recombinarea canalelor
    base_layer = cv2.merge((B_base, G_base, R_base))

    #calcul strat detaliu
    detail_layer = input_image.astype(np.float64) - base_layer


    #imparte fiecare pixel la val maxima, ca sa obtinem valori intre 0 si 1
    #zona cu margini => val aproape de 1
    #zone netede => val aproape de 0
    eta = gamma_map / np.max(gamma_map)
    #este in functie de gamma map (daca gamma e mare, atunci avem o zona cu detalii)
    #transformam harta in format color, pe 3 canale
    eta_color = np.repeat(eta[:, :, None], 3, axis=2)


    #calculam imaginea imbunatatita
    #enhanced = imagine de input + (harta ponderata x strat de detalii x amplificare)
    enhanced = input_image.astype(np.float64) + eta_color * detail_layer * theta
    #theta e amplificarea detaliilor


    #limitarea valorilor la 0-255
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    base_display = np.clip(base_layer, 0, 255).astype(np.uint8)

    #afisarea imaginilor in panouri
    show_image(base_display, panel_base_lbl)
    show_image(enhanced, panel_enhanced_lbl)
    output_image = enhanced



def show_image(img, panel_label):
    # Convertim la RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    # Aflăm dimensiunea panoului
    panel_label.update_idletasks()  # Actualizează info despre panou
    panel_w = panel_label.winfo_width()  # Lățimea panoului
    panel_h = panel_label.winfo_height()  # Înălțimea panoului

    # Dacă panoul e foarte mic (la început), punem niște valori minime
    if panel_w < 50: panel_w = 500
    if panel_h < 50:
        panel_h=500

    # Folosim thumbnail care păstrează aspect ratio și nu depășește cutia
    img_pil.thumbnail((panel_w, panel_h), Image.LANCZOS)

    img_tk = ImageTk.PhotoImage(img_pil)

    panel_label.configure(image=img_tk, text="")  # Scoatem textul dacă e poză
    panel_label.image = img_tk


def clear_panel(panel, text):
    panel.configure(image='', text=text)


# LAYOUT STRUCTURAT (DOCKING)

#container pentru butoane si slidere
#ZONA DE JOS (CONTROALE)
frame_controls_container = Frame(root, bg=COLORS["bg_panel"])
frame_controls_container.pack(side=BOTTOM, fill=X)

# --- Interiorul zonei de jos ---
# Butoane
frame_actions = Frame(frame_controls_container, bg=COLORS["bg_panel"], pady=10)
frame_actions.pack(side=TOP, fill=X)

btn_style = {"font": ("Segoe UI", 11, "bold"), "fg": "white", "bd": 0, "padx": 20, "pady": 5, "cursor": "hand2"}

btn_open = Button(frame_actions, text="Incarca o imagine", command=open_image, bg=COLORS["accent_1"],
                  activebackground="#2980b9", **btn_style)
btn_open.pack(side=LEFT, padx=20, expand=True)

btn_process = Button(frame_actions, text="Filtreaza", command=enhance_image, bg=COLORS["accent_2"],
                     activebackground="#2ecc71", **btn_style)
btn_process.pack(side=LEFT, padx=20, expand=True)

# Slidere
frame_sliders = Frame(frame_controls_container, bg=COLORS["bg_panel"], pady=10)
frame_sliders.pack(side=BOTTOM, fill=X)

slider_style = {
    "bg": COLORS["bg_panel"], "fg": COLORS["text"],
    "highlightthickness": 0, "troughcolor": COLORS["slider_trough"],
    "activebackground": COLORS["accent_1"], "orient": HORIZONTAL,
    "length": 250, "font": ("Segoe UI", 9)
}

# Wrapper pentru centrarea sliderelor
grid_frame = Frame(frame_sliders, bg=COLORS["bg_panel"])
grid_frame.pack()

Label(grid_frame, text="Theta (Detalii)", bg=COLORS["bg_panel"], fg="#cccccc").grid(row=0, column=0, padx=10)
slider_theta = Scale(grid_frame, from_=1, to=40, resolution=0.1, **slider_style)
slider_theta.set(4)
slider_theta.grid(row=1, column=0, padx=10)

Label(grid_frame, text="Raza (r)", bg=COLORS["bg_panel"], fg="#cccccc").grid(row=0, column=1, padx=10)
slider_r = Scale(grid_frame, from_=3, to=40, resolution=1, **slider_style)
slider_r.set(15)
slider_r.grid(row=1, column=1, padx=10)

Label(grid_frame, text="Gamma (Smooth)", bg=COLORS["bg_panel"], fg="#cccccc").grid(row=0, column=2, padx=10)
slider_gamma = Scale(grid_frame, from_=0.1, to=3.0, resolution=0.1, **slider_style)
slider_gamma.set(1.0)
slider_gamma.grid(row=1, column=2, padx=10)

# 2. ZONA DE SUS (TITLU)
header_frame = Frame(root, bg=COLORS["bg_main"])
header_frame.pack(side=TOP, fill=X, pady=10)
Label(header_frame, text="Filtrare Ghidata Ponderata", font=("Segoe UI", 20, "bold"),
      bg=COLORS["bg_main"], fg=COLORS["text"]).pack()

# 3. ZONA DE MIJLOC (IMAGINI)
# Aceasta primește tot spațiul rămas (expand=True)
frame_images_container = Frame(root, bg=COLORS["bg_main"])
frame_images_container.pack(side=TOP, fill=BOTH, expand=True, padx=20, pady=5)


# Funcție helper pentru panouri
def create_image_panel(parent, title):
    # Rama exterioară
    f_out = LabelFrame(parent, text=title, font=("Segoe UI", 10, "bold"),
                       bg=COLORS["bg_main"], fg=COLORS["accent_1"], bd=1, relief=SOLID)
    f_out.pack(side=LEFT, fill=BOTH, expand=True, padx=5, pady=5)

    # Label-ul unde stă poza (centrat)
    lbl = Label(f_out, bg=COLORS["bg_panel"], text="No Image", fg="#666666")
    lbl.pack(fill=BOTH, expand=True, padx=2, pady=2)
    return lbl

#cele 3 panouri: original, strat de baza, imagine imbunatatita
panel_original_lbl = create_image_panel(frame_images_container, "Imagine originala ")
panel_base_lbl = create_image_panel(frame_images_container, "Strat de baza")
panel_enhanced_lbl = create_image_panel(frame_images_container, "Amplificare detalii ")


root.mainloop()