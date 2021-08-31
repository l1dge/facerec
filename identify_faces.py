import face_recognition
import os
from PIL import Image, ImageDraw

unknown_faces = os.listdir("./frec/unknown/")
lee_image = face_recognition.load_image_file("./frec/known/Lee.jpeg")
lee_encoding = face_recognition.face_encodings(lee_image)[0]
Bob_image = face_recognition.load_image_file("./frec/known/Bob.jpeg")
Bob_encoding = face_recognition.face_encodings(Bob_image)[0]
boris_image = face_recognition.load_image_file("./frec/known/boris.jpeg")
boris_encoding = face_recognition.face_encodings(boris_image)[0]
Donald_image = face_recognition.load_image_file("./frec/known/Donald.jpeg")
Donald_encoding = face_recognition.face_encodings(Donald_image)[0]
Julian_image = face_recognition.load_image_file("./frec/known/Julian.jpeg")
Julian_encoding = face_recognition.face_encodings(Julian_image)[0]
MarkW_image = face_recognition.load_image_file("./frec/known/MarkW.jpeg")
MarkW_encoding = face_recognition.face_encodings(MarkW_image)[0]
Todd_image = face_recognition.load_image_file("./frec/known/Todd.jpeg")
Todd_encoding = face_recognition.face_encodings(Todd_image)[0]
Lisa_image = face_recognition.load_image_file("./frec/known/Lisa.jpeg")
Lisa_encoding = face_recognition.face_encodings(Lisa_image)[0]

known_face_encodings = [
    lee_encoding,
    Bob_encoding,
    boris_encoding,
    Donald_encoding,
    Julian_encoding,
    MarkW_encoding,
    Todd_encoding,
    Lisa_encoding,
]

known_face_names = [
    "Lee",
    "Bob",
    "Boris",
    "Donald",
    "Julian",
    "MarkW",
    "Todd",
    "Lisa",
]

for ukface in unknown_faces:
    ukimage = face_recognition.load_image_file(f"./frec/unknown/{ukface}")
    ukface_locations = face_recognition.face_locations(ukimage)
    ukface_encodings = face_recognition.face_encodings(ukimage, ukface_locations,num_jitters=5,model='large')

    # Convert to PIL format
    pil_image = Image.fromarray(ukimage)

    # Set up drawing on image
    draw = ImageDraw.Draw(pil_image)
    for (top, right, bottom, left), ukface_encoding in zip(
        ukface_locations, ukface_encodings
    ):
        matches = face_recognition.compare_faces(known_face_encodings, ukface_encoding)

        name = "Unknown Person"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Draw Box
        draw.rectangle(
            ((left - 10, top - 10), (right + 10, bottom + 10)), outline=(227, 236, 75)
        )

        # Draw Label
        text_width, text_height = draw.textsize(name)
        draw.rectangle(
            ((left - 10, bottom - text_height + 2), (right + 10, bottom + 10)),
            fill=(227, 236, 75),
            outline=(227, 236, 75),
        )
        draw.text((left, bottom - text_height + 5), name, fill=(0, 0, 0, 0))

    del draw
    pil_image.save(f"./frec/identified/{ukface}_scanned.jpg")
    # pil_image.show()
