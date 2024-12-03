import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = "Anthony.jpeg"

def detect_red_objects(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Gambar tidak ditemukan.")
        return

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red_1 = np.array([0, 120, 70])
    upper_red_1 = np.array([10, 255, 255])
    lower_red_2 = np.array([170, 120, 70])
    upper_red_2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv_image, lower_red_1, upper_red_1)
    mask2 = cv2.inRange(hsv_image, lower_red_2, upper_red_2)
    red_mask = mask1 + mask2

    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    result_image = image.copy()
    for contour in contours:
        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(result_image, "Red Object", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title("Gambar Asli")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(result_image_rgb)
    plt.title("Deteksi Objek Merah")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

detect_red_objects(image_path)

def detect_edges(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Gambar tidak ditemukan.")
        return

    edges = cv2.Canny(image, threshold1=100, threshold2=200)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Gambar Asli")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(edges, cmap='gray')
    plt.title("Deteksi Tepi")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

detect_edges(image_path)

def show_histogram(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Gambar tidak ditemukan.")
        return

    colors = ('b', 'g', 'r')
    plt.figure(figsize=(10, 5))
    for i, color in enumerate(colors):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    plt.title("Histogram Warna")
    plt.xlabel("Intensitas Warna")
    plt.ylabel("Jumlah Piksel")
    plt.show()

show_histogram(image_path)

def calculate_object_area(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Gambar tidak ditemukan.")
        return

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red_1 = np.array([0, 120, 70])
    upper_red_1 = np.array([10, 255, 255])
    lower_red_2 = np.array([170, 120, 70])
    upper_red_2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv_image, lower_red_1, upper_red_1)
    mask2 = cv2.inRange(hsv_image, lower_red_2, upper_red_2)
    red_mask = mask1 + mask2

    area = cv2.countNonZero(red_mask)
    print(f"Luas objek merah: {area} piksel")

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Gambar Asli")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(red_mask, cmap='gray')
    plt.title("Mask Objek Merah")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

calculate_object_area(image_path)

