import os
import tensorflow as tf
import matplotlib.pyplot as plt
import cifar10_input

def read_svhn(data_dir="", batch_size = 1, shuffle = True):
    # TODO: load label and return it too

    # create batch
    temp_label = tf.constant(5, dtype=tf.int32)
    batch = cifar10_input._generate_image_and_label_batch(load_image(), temp_label,1, 3, 1)

    return batch


def load_image_batch(image_queue):
    # read in all the the pngs using the queue runner, augment them on the fly.
    # convert the .mat file to a csv and read that as well.
    print("Loading image")

    # print(tf.train.match_filenames_once('./images/*.jpg'))
    #fns = tf.train.match_filenames_once("./images/*.png")


    #fn_queue = tf.train.slice_input_producer([fns], shuffle = True)
    #fn_queue = tf.train.string_input_producer(fns, shuffle=True)

    #img_queue = tf.read_file(fn_queue)  # read image data from file name TODO: MAKE SURE THAT ALL FILES IN THE BATCH ARE READ AND NOT ONLY A SINGLE IMAGE!!!
    img_queue = tf.image.decode_png(image_queue, channels=3, dtype=tf.uint8)

    img_queue = tf.image.resize_image_with_crop_or_pad(img_queue, 400, 400)

    img_batch = tf.train.shuffle_batch([img_queue], batch_size=4, capacity = 20, min_after_dequeue=1 )
    #
    return img_batch# , labels


def create_image_queue():
    fns = tf.train.match_filenames_once("./images/*.png")
    #fn_queue = tf.train.slice_input_producer([fns], shuffle=True)
    fn_queue = tf.train.string_input_producer(fns, num_epochs = 6, shuffle=True)
    reader = tf.WholeFileReader()
    #img_queue = tf.read_file(fns)
    key, img_queue = reader.read(fn_queue)
    return img_queue #fn_queue


def main(argv=None):
    print("Starting...")

    img_queue = load_image_batch(create_image_queue())
    #batch_op = read_svhn()
    #test_op = load_image()
    init = (tf.global_variables_initializer(), tf.local_variables_initializer())

    # Start a new session to show example output.
    with tf.Session() as sess:
        # Required to get the filename matching to run.

        sess.run(init)
        print("Initialized variables...")

        # Coordinate the loading of image files.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # Get an image tensor and print its value.
        image_tensor = sess.run(img_queue)
        print("Image tensor shape:", image_tensor.shape)

        # fns = sess.run(test_op)
        #print("Files after SIP:", fns)

        #plt.imshow(image_tensor)
        #plt.waitforbuttonpress()

        # OPT: resize image?

        # OPT: normalize image in some way?

        # OPT: change image dtype?
        #   reshaped_image = tf.cast(read_input.uint8image, tf.float32)

        # resize image to [batch_id h w channels] ?

        #batch = sess.run(batch_op)
        #print(batch[0][0])
        #plt.imshow(batch[0][0])
        #for img in batch[0]:
        #    print(img)
        #    plt.figure()
        #    plt.imshow(img)
        #    #plt.waitforbuttonpress()

        # Finish off the filename queue coordinator.
        coord.request_stop()
        coord.join(threads)


if __name__ == "__main__":
    main()