# from math import degrees, radians, sin, asin, cos, acos, tan, atan, sqrt, pi
import numpy as np
# import re


def peak_pos(x, y, schwelle=0.5):
    """Funktion zum Berechnen der linken-rechten Kante und der Schrittweite. Franks IDL-
    peak_pos Prozedur, nur reduziert. Daraus lässt sich die FWHM bestimmen.

    :param x x-data  muss Liste sein!
    :param y y-data  muss Liste sein!
    :param schwelle treshold for centre, e.g. 0.5 for 50%
    """

    # if len(x) != len(y):
    #     return

    links = False
    rechts = False
    deltax = False

    # wenn das y-Maximum negativ ist, multipliziere alle y-Werte mit -1
    if max(y) < 0:
        y = y * -1

    sp = np.vstack((x, y)).T

    sp = sp[~np.isnan(sp).any(axis=1)]  # lösche alle Zeilen mit nicht numerischen Einträgen, NaN z.B.

    n_el = len(sp)

    # Abschneiden hoher Werte am Anfang
    anfang = 0
    if n_el > 2:
        while sp[anfang, 1] > sp[anfang + 1, 1] and anfang < n_el - 2:
            anfang = anfang + 1

    sp = sp[anfang::, :]
    n_el = n_el - anfang

    # # if not enough data points or signal too low --> return
    # total_signal = sum(row[1] for row in sp)
    # if n_el < 5 or total_signal < 1e-15:
    #     return links, rechts, deltax

    # if not enough data points --> return
    if n_el < 4:
        return links, rechts, deltax

    p = max(sp[:, 1])
    nmax = n_el - 1
    index = np.argmax(sp[:, 1])
    schw = p * schwelle
    # mi = max(min(sp[0:index, 1]), min(sp[index:, 1]))  # das hoehere der beiden Minima rechts und links des Max

    i1 = index
    i2 = index

    while sp[i1, 1] > schw and i1 > 0:
        i1 = i1 - 1
    if i1 != index:  # Abfangen Fall: Index ist letzter Wert im Array
        i1 = i1 + 1  # kanal vor der schwelle

    while sp[i2, 1] > schw and i2 < nmax:
        i2 = i2 + 1
    if i2 - 1 > 1:  # Abfangen von neg. Index
        i2 = i2 - 1
    else:
        i2 = 1

    # bearbeiten des linken Randes
    deltax = sp[i1, 0] - sp[i1 - 1, 0]
    deltay = sp[i1, 1] - sp[i1 - 1, 1]
    if deltay != 0:
        shft = (schw - sp[i1 - 1, 1]) / deltay
        links = sp[i1 - 1, 0] + shft * deltax

    # bearbeiten rechter Rand
    deltax = sp[i2 + 1, 0] - sp[i2, 0]
    deltay = sp[i2 + 1, 1] - sp[i2, 1]
    if deltay != 0:
        shft = (schw - sp[i2 + 1, 1]) / deltay
        rechts = sp[i2 + 1, 0] + shft * deltax

    # wenn eine der Kanten innerhalb und die andere außerhalb des Scan-Bereichs liegt, deutet das auf einen Dip hin.
    # führe die Rechnung noch einmal mit der Suchen nach einem Minimum durch:

    if sp[0, 0] < links < sp[-1, 0] and not sp[0, 0] < rechts < sp[-1, 0] or \
            sp[0, 0] < rechts < sp[-1, 0] and not sp[0, 0] < links < sp[-1, 0]:

        links = False
        rechts = False
        deltax = False

        # y = y * -1

        sp = np.vstack((x, y)).T

        sp = sp[~np.isnan(sp).any(axis=1)]  # lösche alle Zeilen mit nicht numerischen Einträgen, NaN z.B.

        n_el = len(sp)

        # Abschneiden hoher Werte am Anfang
        anfang = 0
        if n_el > 2:
            while sp[anfang, 1] > sp[anfang + 1, 1] and anfang < n_el - 2:
                anfang = anfang + 1

        sp = sp[anfang::, :]
        n_el = n_el - anfang

        p = min(sp[:, 1])
        nmax = n_el - 1
        index = np.argmin(sp[:, 1])
        schw = p * 1/schwelle  # 1/schwelle weil es hier nach dem Minimum geht
        # mi = max(min(sp[0:index, 1]), min(sp[index:, 1]))  # das hoehere der beiden Minima rechts und links des Max

        i1 = index
        i2 = index

        while sp[i1, 1] < schw and i1 > 0:
            i1 = i1 - 1
        if i1 != index:  # Abfangen Fall: Index ist letzter Wert im Array
            i1 = i1 + 1  # kanal vor der schwelle

        while sp[i2, 1] < schw and i2 < nmax:
            i2 = i2 + 1
        if i2 - 1 > 1:  # Abfangen von neg. Index
            i2 = i2 - 1
        else:
            i2 = 1

        # bearbeiten des linken Randes
        deltax = sp[i1, 0] - sp[i1 - 1, 0]
        deltay = sp[i1, 1] - sp[i1 - 1, 1]
        if deltay != 0:
            shft = (schw - sp[i1 - 1, 1]) / deltay
            links = sp[i1 - 1, 0] + shft * deltax

        # bearbeiten rechter Rand
        deltax = sp[i2 + 1, 0] - sp[i2, 0]
        deltay = sp[i2 + 1, 1] - sp[i2, 1]
        if deltay != 0:
            shft = (schw - sp[i2 + 1, 1]) / deltay
            rechts = sp[i2 + 1, 0] + shft * deltax

    return links, rechts, deltax


def search_for_curve(x_array, y_array):

    """Filtert aus einem Plot eine Kurve heraus. Geht vom Maximum los und holt alle Datenpunkte bis 1% vom Maximum. """

    maximum_index = y_array.argmax()

    linke_seite_y = np.array([])
    linke_seite_x = np.array([])

    linke_seite_y = np.append(linke_seite_y, y_array[maximum_index])
    linke_seite_x = np.append(linke_seite_x, x_array[maximum_index])

    rechte_seite_y = np.array([])
    rechte_seite_x = np.array([])

    # linke Seite
    i = maximum_index
    while y_array[i - 1] >= y_array[maximum_index]*0.2:
        linke_seite_y = np.append(linke_seite_y, y_array[i - 1])
        linke_seite_x = np.append(linke_seite_x, x_array[i - 1])
        i -= 1

    i = maximum_index

    # rechte Seite
    while y_array[i + 1] >= y_array[maximum_index]*0.2:
        rechte_seite_y = np.append(rechte_seite_y, y_array[i + 1])
        rechte_seite_x = np.append(rechte_seite_x, x_array[i + 1])
        i += 1

    linke_seite_y = np.flip(linke_seite_y)
    linke_seite_x = np.flip(linke_seite_x)

    y_array = np.append(linke_seite_y, rechte_seite_y)
    x_array = np.append(linke_seite_x, rechte_seite_x)

    return x_array, y_array


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def isint(value):
    try:
        int(value)
        return True
    except ValueError:
        return False
