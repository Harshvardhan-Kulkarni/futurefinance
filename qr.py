import qrcode

def generate_qr_code(ip_address):
    # Create QR code instance
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    
    # Add data (in this case, the IP address)
    qr.add_data(ip_address)
    qr.make(fit=True)

    # Create an image from the QR code instance
    img = qr.make_image(fill_color="black", back_color="white")

    # Save or display the image
    img.save("qrcode.png")  # Save the image to a file
    img.show()  # Display the image using the default image viewer

if __name__ == "__main__":
    # Your main code logic here

    # Replace '192.168.1.1' with your actual IP address
    ip_address = "http://98.70.56.152:8501/"
    generate_qr_code(ip_address)