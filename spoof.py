"""
spoof.py
===================================
This module creates modified copies of a spectrogram which can be
saved as PNGs.
"""

import os
from pathlib import Path
from PIL import Image
import numpy as np
import random
import ntpath


class SpoofedData:
    """
    This class creates modified copies of the source spectrogram and which can be
    saved as PNGs. Possible modifications include, randomly adjusting the amplitude,
    time-shifting, and adding white noise.

    Parameters
    ----------
    filepath: string
        An absolute or relative path to a grayscale spectrogram png.
    iters: int
        The number of spoofed spectrograms to be created.
    """
    VOL_ADJUST_RANGE = 50  #: the most (+/-) the amplitude(volume) can be adjusted
    RANDOM_NOISE_CHANCE = 10  #: the chance (1/x) that a pixel becomes random noise
    AMP_FREQ_RANGE = 80  #: the most (+/-) the frequency bands can be adjusted
    AMP_TIME_RANGE = 80  #: the most (+/-) the time bands can be adjusted
    MAX_AMP_FREQ = 5  #: the maximum fraction (1/x) of frequency bands to be adjusted
    MAX_AMP_TIME = 5  #: the maximum fraction (1/x) of time bands to be adjusted

    def __init__(self, filepath, iters):
        """
        __init__

        Initializes an instance of SpoofedData

        Parameters
        ----------
        filepath: string
            An absolute or relative path to a grayscale spectrogram png.
        iters: int
            The number of spoofed spectrograms to be created.
        """

        self.path = Path(filepath)
        # load the image
        self.image = Image.open(self.path)
        # convert image to numpy array
        self.og_data = np.asarray(self.image)
        self.iterations = iters
        self.new_data = []
        # create copies to modify
        for x in range(self.iterations):
            self.new_data.append(np.copy(self.og_data))

    def adjust_volume(self):
        """
        adjust_volume

        This function adjusts the amplitude ("volume") of the spectrogram by
        a random amount. This is accomplished by adjusting the grayscale
        color of each pixel.
        """
        for img in range(0, self.iterations):
            arr = self.new_data[img]
            delta = random.randrange(-self.VOL_ADJUST_RANGE, self.VOL_ADJUST_RANGE, 1)
            # create an array of deltas to combine with the existing array
            vol = np.full((self.og_data.shape[0], self.og_data.shape[1],
                           self.og_data.shape[2]), [delta, delta, delta, 0])
            # bounded sum of the two arrays
            for x in range(len(arr)):
                result = arr[x] + vol[x]
                result = np.clip(result, 0, 255)
                arr[x] = result

            self.new_data[img] = arr
    
    def time_shift(self):
        """
        time_shift

        This function shifts the spectrogram with respect to time by a random amount.
        The spectrogram "rolls" so that no data is lost. For example, data at the right
        side of the spectrogram may have shifted around to the left. This function also
        smooths in any empty space at the beginning or ending of the recording.
        """
        for img in range(0, self.iterations):
            arr = self.new_data[img]
            # if the number used is not 4, the spectrograms may not be grayscale
            shift = random.randrange(0, self.og_data.shape[1], 4)
            arr = self.fill_empty_borders(arr)
            arr = np.roll(arr, shift)
            self.new_data[img] = arr

    def random_noise(self):
        """
        random_noise

        This function adds in random "white noise" to the spectrogram by setting a
        random percentage of pixels to random grayscale values.
        """
        for img in range(0, self.iterations):
            arr = self.new_data[img]
            for x in range(self.og_data.shape[0]):
                for y in range(self.og_data.shape[1]):
                    # (1 / RANDOM_NOISE_CHANCE) that the data is randomized at that location
                    if random.randrange(1, self.og_data.shape[0] * self.og_data.shape[1] *
                                           self.og_data.shape[2], 1) % self.RANDOM_NOISE_CHANCE == 0:
                        z = random.randrange(0, 255, 1)
                        arr[x][y] = [z, z, z, 255]
            self.new_data[img] = arr

    def background_noise(self, filepath):
        """
        background_noise

        This function adds in a random "background noise" from a user-specified directory.
        This is accomplished by creating a spectrogram that has the average values of the
        two source spectrograms at each location

        Parameters
        ----------
        filepath: string
            An absolute or relative path to a grayscale spectrogram png.
        """

        filepath = Path(filepath)
        for img in range(0, self.iterations):
            arr = self.new_data[img]
            background = random.choice(os.listdir(filepath))
            background_img = Image.open(filepath + background)
            background_arr = np.asarray(background_img)
            self.new_data[img] = self.combine_arr(arr, background_arr)

    def combine_arr(self, arr1, arr2):
        """
        combine_arr

        This function returns an array that is the average of two source arrays
        at each location. The arrays must have the same shape.

        Parameters
        ----------
        arr1: numpy array
            the first array to be averaged
        arr2: numpy array
            the second array to be averaged
        """
        result = np.copy(self.og_data)
        for x in range(len(arr1)):
            # numpy does not like this operation if done with the whole
            # array, so parts of the arrays are used.
            result[x] = arr1[x] / 2 + arr2[x] / 2
        return result

    def fill_empty_borders(self, data):
        """
        fill_empty_borders

        This function fills in the empty vertical borders of a numpy array if
        they are empty. It does so by copying the nearest nonempty column.

        Parameters
        ----------
        data
            a numpy array with borders to be filled
        """
        empty = np.full((self.og_data.shape[1], 1, self.og_data.shape[2]), 255)
        # check if far right is empty
        if np.all(data[:, self.og_data.shape[1] - 1] == empty):
            not_empty = self.og_data.shape[1] - 1
            # determines the closest non-empty column
            while np.all(data[:, not_empty] == empty) and not_empty >= 0:
                not_empty -= 1
            # fills in empty columns with the filled columns values
            if not_empty >= 0:
                if not_empty > 0:
                    not_empty -= 1
                filled = data[:, not_empty]
                while not_empty < self.og_data.shape[1] - 1:
                    data[:, not_empty] = filled
                    not_empty += 1
        # check if far left is empty
        if np.all(data[:, 0] == empty):
            not_empty = 0
            # determines the closest non-empty column
            while np.all(data[:, not_empty] == empty) and not_empty < self.og_data.shape[1] - 1:
                not_empty += 1
            # fills in empty columns with the filled columns values
            if not_empty <= self.og_data.shape[1] - 1:
                if not_empty < self.og_data.shape[1] - 1:
                    not_empty += 1
                filled = data[:, not_empty]
                while not_empty >= 0:
                    data[:, not_empty] = filled
                    not_empty -= 1
        return data

    def save_image(self, dest_path=None, replace=False):
        """
        save_image

        This function saves the newly created numpy arrays as grayscale
        spectrogram pngs. A directory to save the images in and whether or not
        to replace the source image can be specified.

        Parameters
        ----------
        dest_path: string
            The directory to place the new images. By default the
            source directory is used.
        replace: bool
            Whether or not the source image should be replaced.
        """
        dest = Path(dest_path)
        for num in range(self.iterations):
            new_image = Image.fromarray(self.new_data[num], "RGBA")

            if dest is not None: # create designated new destination path and filename
                img_dest = str(dest) + '\\' + ntpath.basename(str(self.path))
            else:
                img_dest = self.path # use source path

            if replace and os.path.exists(Path(img_dest)):
                os.remove(Path(img_dest))
            # only include an added number if more than one new image is being saved
            if self.iterations != 1:
                img_dest = img_dest[0:len(img_dest) - 4]
                img_dest += '_' + str(num) + '.png'

            # remove any previously spoofed data with the same name at the desitantion
            if replace and os.path.exists(Path(img_dest)):
                os.remove(Path(img_dest))

            new_image.save(img_dest)

    def amp_freq(self):
        """
        amp_freq

        This function creates numpy arrays representing spectrograms where a
        random number of frequency bands have had their amplitudes change by a
        random quantity.
        """
        for img in range(0, self.iterations):
            arr = self.new_data[img]
            # how many bands to modify
            bands = random.randrange(0, int(self.og_data.shape[0] / self.MAX_AMP_FREQ), 1)
            # how much to modify the bands by
            amplitude = random.randrange(-self.AMP_FREQ_RANGE, self.AMP_FREQ_RANGE, 1)
            # which bands to modify
            loc = random.randrange(0, self.og_data.shape[0] - bands, 1)
            for x in range(loc, loc + bands):
                for y in range(self.og_data.shape[1]):
                    arr[y] = self.change_amp(arr[y], amplitude)

            self.new_data[img] = arr

    def amp_time(self):
        """
        amp_time

        This function creates numpy arrays representing spectrograms where a
        random number of time bands have had their amplitudes change by a
        random quantity.
        """
        for img in range(0, self.iterations):
            arr = self.new_data[img]
            # how many bands to modify
            bands = random.randrange(0, int(self.og_data.shape[1] / self.MAX_AMP_TIME), 1)
            # how much to modify the bands by
            amplitude = random.randrange(-self.AMP_TIME_RANGE, self.AMP_TIME_RANGE, 1)
            # which bands to modify
            loc = random.randrange(0, self.og_data.shape[1] - bands, 1)
            for y in range(loc, loc + bands):
                for x in range(self.og_data.shape[0]):
                    arr[x] = self.change_amp(arr[x], amplitude)

            self.new_data[img] = arr

    @staticmethod
    def change_amp(arr, amplitude):
        """
        change_amp

        This function modifies the RGB values stored in a numpy array by a
        specified amount.

        Parameters
        ----------
        arr
            The array which will have its values modified
        amplitude
            The amount to modify the values in arr by
        """
        for z in range(3):
            if arr[z] + amplitude > 255:
                arr[z] = 255
            elif arr[z] + amplitude < 0:
                arr[z] = 0
            else:
                arr[z] += amplitude
        return arr


# Function for manually testing the code with a specific input
def main():
    # In this example a SpoofedData instance is created with a specified filepath
    # and 20 is input as the number of spectrograms to create. Each subsequent call
    # after the initialization to the classes functions, modifies the new spectrograms
    # in a certain way. Random number generators ensure that each one is different.
    # Finally, the save_image function is called with a specified destination directory
    # and the 20 new spectrograms are saved.
    file = 'C:\\Users\\Leo Glikbarg\\PycharmProjects\\bird\\Birdsong_Spectrograms\\Hylophilusdecurtatus67760.png'
    dest = 'C:\\Users\\Leo Glikbarg\\PycharmProjects\\bird\\test'
    background_noises = 'C:\\Users\\Leo Glikbarg\\PycharmProjects\\bird\\Background_Sounds'

    sd = SpoofedData(file, 20)
    sd.time_shift()
    sd.adjust_volume()
    sd.background_noise(background_noises)
    sd.save_image(dest)

    print("\nSpoofed data created and placed in " + str(dest))
    exit(0)


# calls main() if this file is run directly
if __name__ == "__main__":
    main()
