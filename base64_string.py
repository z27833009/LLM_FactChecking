def base64string(s):
    with open(images_path, "rb") as image_file:
        # Convert images to Base64 encoding
        encoded_string = base64.b64encode(image_file.read()).decode()
        print(encoded_string)
        
# print(base64string("images_path"))