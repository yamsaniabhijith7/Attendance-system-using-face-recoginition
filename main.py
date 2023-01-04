from sklearn.metrics.pairwise import pairwise_distances
from tensorflow.python.platform import gfile
import tensorflow as tf
import numpy as np
import detect_and_align
import argparse
import time
import cv2
import os
import smtplib,ssl
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase 
from email.mime.text import MIMEText 
from email.utils import formatdate
from email import encoders
from datetime import date

class IdData:
    """Keeps track of known identities and calculates id matches"""

    def __init__(
        self, id_folder, mtcnn, sess, embeddings, images_placeholder, phase_train_placeholder, distance_treshold
    ):
        print("Loading known identities: ", end="")
        self.distance_treshold = distance_treshold
        self.id_folder = id_folder
        self.mtcnn = mtcnn
        self.id_names = []

        image_paths = []
        ids = os.listdir(os.path.expanduser(id_folder))
        for id_name in ids:
            id_dir = os.path.join(id_folder, id_name)
            image_paths = image_paths + [os.path.join(id_dir, img) for img in os.listdir(id_dir)]

        print("Found %d images in id folder" % len(image_paths))
        aligned_images, id_image_paths = self.detect_id_faces(image_paths)
        feed_dict = {images_placeholder: aligned_images, phase_train_placeholder: False}
        self.embeddings = sess.run(embeddings, feed_dict=feed_dict)

        if len(id_image_paths) < 5:
            self.print_distance_table(id_image_paths)

    def detect_id_faces(self, image_paths):
        aligned_images = []
        id_image_paths = []
        for image_path in image_paths:
            image = cv2.imread(os.path.expanduser(image_path), cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_patches, _, _ = detect_and_align.detect_faces(image, self.mtcnn)
            if len(face_patches) > 1:
                print(
                    "Warning: Found multiple faces in id image: %s" % image_path
                    + "\nMake sure to only have one face in the id images. "
                    + "If that's the case then it's a false positive detection and"
                    + " you can solve it by increasing the thresolds of the cascade network"
                )
            aligned_images = aligned_images + face_patches
            id_image_paths += [image_path] * len(face_patches)
            path = os.path.dirname(image_path)
            self.id_names += [os.path.basename(path)] * len(face_patches)

        return np.stack(aligned_images), id_image_paths

    def print_distance_table(self, id_image_paths):
        """Prints distances between id embeddings"""
        distance_matrix = pairwise_distances(self.embeddings, self.embeddings)
        image_names = [path.split("/")[-1] for path in id_image_paths]
        print("Distance matrix:\n{:20}".format(""), end="")
        [print("{:20}".format(name), end="") for name in image_names]
        for path, distance_row in zip(image_names, distance_matrix):
            print("\n{:20}".format(path), end="")
            for distance in distance_row:
                print("{:20}".format("%0.3f" % distance), end="")
        print()

    def find_matching_ids(self, embs):
        matching_ids = []
        matching_distances = []
        distance_matrix = pairwise_distances(embs, self.embeddings)
        for distance_row in distance_matrix:
            min_index = np.argmin(distance_row)
            if distance_row[min_index] < self.distance_treshold:
                matching_ids.append(self.id_names[min_index])
                matching_distances.append(distance_row[min_index])
            else:
                matching_ids.append(None)
                matching_distances.append(None)
        return matching_ids, matching_distances


def load_model(model):
    model_exp = os.path.expanduser(model)
    if os.path.isfile(model_exp):
        print("Loading model filename: %s" % model_exp)
        with gfile.FastGFile(model_exp, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name="")
    else:
        raise ValueError("Specify model file, not directory!")


def main(args):
    with tf.Graph().as_default():
        with tf.Session() as sess:

            # Setup models
            mtcnn = detect_and_align.create_mtcnn(sess, None)

            load_model(args.model)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Load anchor IDs
            id_data = IdData(
                args.id_folder[0],
                mtcnn,
                sess,
                embeddings,
                images_placeholder,
                phase_train_placeholder,
                args.threshold,
            )

            # OPEN CAMERA AND TAKE A SNAPSHOT
            cam = cv2.VideoCapture(0)
            cv2.namedWindow("test")
            img_counter = 0

            while True:
                ret, frame = cam.read()
                cv2.imshow("test", frame)
                if not ret:
                    break
                k = cv2.waitKey(1)

                if k%256 == 27:
                    # ESC pressed
                    print("Escape hit, closing...")
                    break
                elif k%256 == 32:
                    # SPACE pressed
                    img_name = "attendance" + str(date.today()) + ".png"
                    cv2.imwrite(img_name, frame)
                    print("{} written!".format(img_name))
                    img_counter += 1

            cam.release()

            cv2.destroyAllWindows()
            
            # Now that we have an image, we detect and recognise the faces and append them in list
            present = []

            face_patches, padded_bounding_boxes, landmarks = detect_and_align.detect_faces(frame, mtcnn)

            if len(face_patches) > 0:
                face_patches = np.stack(face_patches)
                feed_dict = {images_placeholder: face_patches, phase_train_placeholder: False}
                embs = sess.run(embeddings, feed_dict=feed_dict)

                print("Attendance:")
                matching_ids, matching_distances = id_data.find_matching_ids(embs)

                for bb, landmark, matching_id, dist in zip(
                    padded_bounding_boxes, landmarks, matching_ids, matching_distances
                ):
                    if matching_id is None:
                        matching_id = "Unknown"
                        print("Unknown! Couldn't find match.")
                    else:
                        print("%s" % (matching_id)) #prints all the names present in the image
                        present.append(matching_id)
                        with open('presentstudents.txt', 'w') as filehandle: 
                            for listitem in present:
                                filehandle.write('%s\n' % listitem)
            else:
                print("Couldn't find a face")
                
def send_an_email():
    subject = "Attendance on date: " + str(date.today())
    body = "This is an email with attachment sent from Python"
    sender_email = input("Enter your email id: ") # "developericewich@gmail.com"
    password = input("Type your password and press enter:") # "developertesting"
    receiver_email = input("Enter receiver's email id: ") # "developericewich@gmail.com"
    
    # Create a multipart message and set headers
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject
    message["Bcc"] = receiver_email  # Recommended for mass emails

    # Add body to email
    message.attach(MIMEText(body, "plain"))

    attendancefile = "presentstudents.txt"  # In same directory as script
    attendancephoto = "attendance" + str(date.today()) + ".png"

    # Open txt file in binary mode
    with open(attendancefile, "rb") as attachment:
        # Add file as application/octet-stream
        # Email client can usually download this automatically as attachment
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())

    # Open png file in binary mode
    with open(attendancephoto, "rb") as attachment:
        # Add file as application/octet-stream
        # Email client can usually download this automatically as attachment
        part2 = MIMEBase("application", "octet-stream")
        part2.set_payload(attachment.read())

    # Encode file in ASCII characters to send by email    
    encoders.encode_base64(part)
    encoders.encode_base64(part2)

    # Add header as key/value pair to attachment part
    part.add_header(
        "Content-Disposition",
        f"attachment; filename= {attendancefile}",
    )
    part2.add_header(
        "Content-Disposition",
        f"attachment; filename= {attendancephoto}",
    )
    # Add attachment to message and convert message to string
    message.attach(part)
    message.attach(part2)
    text = message.as_string()

    # Log in to server using secure context and send email
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, text)
    print("Email sent to " + receiver_email)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("model", type=str, help="Path to model protobuf (.pb) file")
    parser.add_argument("id_folder", type=str, nargs="+", help="Folder containing ID folders")
    parser.add_argument("-t", "--threshold", type=float, help="Distance threshold defining an id match", default=1.2)
    main(parser.parse_args())
    send_an_email()