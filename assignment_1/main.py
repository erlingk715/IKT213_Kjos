import cv2

def print_image_information(image):
    # Hent høyde, bredde og antall kanaler
    height, width, channels = image.shape

    print("Height:", height)
    print("Width:", width)
    print("Channels:", channels)
    print("Size (number of values):", image.size)
    print("Data type:", image.dtype)


if __name__ == "__main__":
    # Les inn bildet lena.png
    image = cv2.imread("lena-1.png")
    if image is None:
        print("Feil: Kunne ikke laste inn bildet!")
    else:
        print_image_information(image)
