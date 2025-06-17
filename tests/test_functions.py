import sys
import os
import pytest
import numpy as np
from PIL import Image
import tkinter as tk
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")
import locale

# Přidáme do sys.path nadřazený adresář aktuálního souboru, aby Python našel modul 'app'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.functions import dicom_image
from app.functions import align_images, posunuti_image
from app.functions import ROI_drawer_manual
from app.functions import premenovy_zakon
from app.functions import tew_correction
from app.functions import compute_time_differences
from app.functions import Graf_1
from app.functions import riu_uptace_fce, riu_fit


# Fixture: zakladni mockovany DICOM objekt
# Tato fixture vraci "mock" objekt, ktery simuluje DICOM soubor pro testovani.
# Pouziva MagicMock z pytestu/pytest-mock, ktery umoznuje definovat vlastnosti a chovani objektu.
@pytest.fixture
def dicom_mock():
    mock = MagicMock()
    # pixel_array je pole numpy, ktere obsahuje 6 vrstev (4x4 matice), kazda vrstva je naplnena hodnotou odpovidajici indexu vrstvy (0 az 5)
    mock.pixel_array = np.array([np.ones((4, 4)) * i for i in range(6)])
    # definuje chovani __getitem__, tedy pristup k DICOM tagum ve stylu dcm[(group, element)]
    # vraci MagicMock s atributem value podle pozadovaneho tagu
    mock.__getitem__.side_effect = lambda x: {
        (0x0008, 0x0022): MagicMock(value="20250101"),  # datum akvizice
        (0x0008, 0x0032): MagicMock(value="101010"),  # cas akvizice
        (0x0018, 0x1242): MagicMock(value=1500),  # doba trvani v ms
    }[x]
    return mock


def test_load_dicom_standard_six_layers(dicom_mock):
    # Testuje spravne nacteni DICOM dat s 6 vrstvami
    # patchujeme pydicom.dcmread, aby vzdy vracel nas mockovany DICOM objekt, nebot nechceme cist skutecny soubor
    with patch("pydicom.dcmread", return_value=dicom_mock):
        dcm = (
            dicom_image()
        )  # vytvori instanci tridy (neni v kodu, ale predpokladame, ze existuje)
        dcm.load_dicom("fake_path.dcm")  # zavola nacteni DICOM dat (vraci nas mock)

        # Kontrola, ze vrstvy jsou nacteny spravne podle mock dat:
        # kazda vrstva by mela byt 4x4 matice plna hodnoty podle indexu vrstvy
        np.testing.assert_array_equal(dcm.ant_pw, np.ones((4, 4)) * 0)
        np.testing.assert_array_equal(dcm.pos_pw, np.ones((4, 4)) * 1)
        np.testing.assert_array_equal(dcm.ant_lsw, np.ones((4, 4)) * 2)
        np.testing.assert_array_equal(dcm.pos_lsw, np.ones((4, 4)) * 3)
        np.testing.assert_array_equal(dcm.ant_usw, np.ones((4, 4)) * 4)
        np.testing.assert_array_equal(dcm.pos_usw, np.ones((4, 4)) * 5)

        # Kontrola spravneho nacteni metadat z DICOM tagu
        assert dcm.acq_date == "20250101"
        assert dcm.acq_time == "101010"
        assert dcm.acq_dur == 1.5  # prevod 1500 ms na 1.5 s

        # Kontrola, ze maxima vrstvy jsou spravne (ant_max a pos_max)
        # ant_pw je nula, proto fallback na 1 (pravdepodobne implementovano v tride)
        assert dcm.ant_max == 1
        assert dcm.pos_max == 1


def test_load_dicom_less_than_6_layers(dicom_mock):
    # Testuje chovani tridy, kdy DICOM data maji mene nez 6 vrstev
    # Zmeni pixel_array na pouze 2 vrstvy
    dicom_mock.pixel_array = np.array([np.ones((4, 4)) * 0, np.ones((4, 4)) * 1])
    with patch("pydicom.dcmread", return_value=dicom_mock):
        dcm = dicom_image()
        dcm.load_dicom("fake_path.dcm")

        # Kontrola, ze fallback vrstvy (ktere nejsou v datech) jsou naplnene jednickami
        # tj. pokud chybi vrstva, ma byt defaultni data 1
        np.testing.assert_array_equal(dcm.ant_lsw, np.ones((4, 4)))
        np.testing.assert_array_equal(dcm.pos_lsw, np.ones((4, 4)))
        np.testing.assert_array_equal(dcm.ant_usw, np.ones((4, 4)))
        np.testing.assert_array_equal(dcm.pos_usw, np.ones((4, 4)))


def test_convert_to_image_valid_types(dicom_mock):
    # Testuje konverzi vrstvy DICOM dat do PIL Image, pokud je zadany platny typ vrstvy
    with patch("pydicom.dcmread", return_value=dicom_mock):
        dcm = dicom_image()
        dcm.load_dicom("fake_path.dcm")

        # Pro vsechny platne typy vrstev zkontroluje, ze vracena hodnota je PIL Image a je v rezimu 'L' (grayscale)
        for img_type in [
            "ant_pw",
            "pos_pw",
            "ant_lsw",
            "pos_lsw",
            "ant_usw",
            "pos_usw",
        ]:
            img = dcm.convert_to_image(img_type)
            assert isinstance(img, Image.Image)
            assert img.mode == "L"  # ocekavany rezim obrazu je grayscale


def test_convert_to_image_invalid_type(dicom_mock):
    # Testuje chovani pri zadani neplatneho typu vrstvy do convert_to_image - ocekava vyjimku
    with patch("pydicom.dcmread", return_value=dicom_mock):
        dcm = dicom_image()
        dcm.load_dicom("fake_path.dcm")

        # Ocekavame, ze pri neplatnem nazvu vrstvy dojde k vyjimce s textem "Invalid planar type"
        with pytest.raises(Exception) as excinfo:
            dcm.convert_to_image("invalid_layer")
        assert "Invalid planar type" in str(excinfo.value)


##### ALIGN A POSUNUTI ----------------------------


def test_align_images_no_shift_needed():
    """
    Testujeme align_images, kdy neni potreba zadny posun.
    Referencni a posouvany obrazek jsou stejne.
    """
    # Vytvorime obrazek 10x10 plny nul, s malym ctvercem o hodnotach 1 uprostred (radky 4-5, sloupce 4-5)
    image = np.zeros((10, 10))
    image[4:6, 4:6] = 1  # maly ctverec uprostred

    # Zavolame funkci align_images s referencnim obrazkem a jeho kopii (takze by nemel byt zadny posun)
    aligned_image, shift_x, shift_y = align_images(image, image.copy(), sigma=0.1)

    # Ocekavame, ze posun v x a y bude nulovy (zadny pohyb nepotrebny)
    assert shift_x == 0
    assert shift_y == 0

    # Zarovnany obrazek musi byt shodny s puvodnim (pixel po pixelu)
    np.testing.assert_array_equal(aligned_image, image)


def test_align_images_with_shift():
    """
    Testujeme align_images, kdy posouvany obrazek je umele posunuty.
    Ocekavame, ze funkce zarovna obrazek zpatky na referencni pozici.
    """
    # Vytvorime referencni obrazek 20x20, s ctvercem 4x4 uprostred (radky 8-11, sloupce 8-11)
    reference = np.zeros((20, 20))
    reference[8:12, 8:12] = 1  # ctverec uprostred

    # Posuneme obrazek o +3 radky (dolu) a -2 sloupce (doleva) pomoci np.roll
    moving = np.roll(reference, shift=(3, -2), axis=(0, 1))

    # Zarovname posunuty obrazek na referencni pozici
    aligned_image, shift_x, shift_y = align_images(reference, moving, sigma=0.1)

    # Ocekavame, ze posun bude opacny k nasemu posunu z np.roll, tedy (2, -3)
    # Pozor na to, ze osa 0 (radky) je posunuta o 3, tedy spravny korekcni posun je -3,
    # ale tady je posun_x a posun_y pravdepodobne definovany opacne, vysledek by mel byt spravny podle implementace
    assert shift_x == 2
    assert shift_y == -3

    # Zarovnany obrazek by mel byt pixelove shodny s referencnim obrazkem
    np.testing.assert_array_equal(aligned_image, reference)


def test_align_images_gaussian_effect():
    """
    Overime, ze zvyseni parametru sigma nezpusobi chybu nebo vyjimku.
    Sigma pravdepodobne ovlivnuje nejakou gaussianu filtraci v align_images.
    """
    # Vytvorime obrazek 15x15 s malym ctvercem 3x3 uprostred
    img1 = np.zeros((15, 15))
    img1[6:9, 6:9] = 1

    # Posuneme obrazek o 1 radek a 1 sloupec
    img2 = np.roll(img1, shift=(1, 1), axis=(0, 1))

    # Zarovname obrazky s vyssim sigma (1.5)
    aligned_image, _, _ = align_images(img1, img2, sigma=1.5)

    # Overime, ze vysledny obrazek ma stejny tvar jako puvodni (15x15)
    assert aligned_image.shape == img1.shape


def test_align_images_invalid_input():
    """
    Testujeme spravne zachazeni s chybou, kdy vstupni obrazky maji rozdilny rozmer.
    Funkce by mela vyhodit vyjimku.
    """
    # Vytvorime dva obrazky s ruznym rozmerem (10x10 a 8x8)
    img1 = np.zeros((10, 10))
    img2 = np.zeros((8, 8))  # jiny rozmer

    # Ocekavame, ze volani align_images s ruznymi rozmery vyhodi Exception
    with pytest.raises(Exception) as excinfo:
        align_images(img1, img2)

    # Ocekavame, ze text vyjimky obsahuje "same shape for alignment"
    assert "same shape for alignment" in str(excinfo.value)


def test_posunuti_image_basic_shift():
    """
    Testujeme jednoduche posunuti obrazku pomoci funkce posunuti_image.
    """
    # Vytvorime obrazek 5x5 s jednickou v pozici (2,2)
    image = np.zeros((5, 5))
    image[2, 2] = 1

    # Posuneme obrazek o 1 pixel doprava (x = 1), bez posunu po ose y
    shifted = posunuti_image(image, shift_x=1, shift_y=0)

    # Ocekavany vysledek je obrazek 5x5 s jednickou na pozici (2,3)
    expected = np.zeros((5, 5))
    expected[2, 3] = 1  # posun doprava

    # Porovname vysledek s ocekavanym
    np.testing.assert_array_equal(shifted, expected)


def test_posunuti_image_wraparound():
    """
    Testujeme chovani posunuti obrazku, kdy posun je mimo hranice a dochazi k wrap-around (ovinuti).
    """
    # Vytvorime obrazek 4x4 s jednickou v levem hornim rohu (0,0)
    image = np.zeros((4, 4))
    image[0, 0] = 1

    # Posuneme obrazek o -1 pixel na ose x i y (tedy nahoru a doleva)
    shifted = posunuti_image(image, shift_x=-1, shift_y=-1)

    # Ocekavame, ze pixel s hodnotou 1 se presune na pozici (3,3) kvuli wrap-around chovani np.roll
    assert shifted[3, 3] == 1
    # Overime, ze soucet hodnot v obrazku zustava 1 (pixel se nikam neztratil)
    assert np.sum(shifted) == 1


#####   ZAKRESLOVANI ROI ---------------------


class DummyDicomObj:
    def __init__(self):
        # Konstruktor vytvori dummy DICOM objekt s prednastavenymi atributy
        # ant_pw a pos_pw jsou 10x10 pole typu float, naplnene konstantnimi hodnotami 5.0 a 10.0
        self.ant_pw = np.ones((10, 10), dtype=float) * 5.0
        self.pos_pw = np.ones((10, 10), dtype=float) * 10.0
        # ROI atributy jsou zacinaji jako None (nemaji zadne vybrane oblasti)
        self.ant_roi = None
        self.pos_roi = None

    def convert_to_image(self, planar_type):
        # Metoda vraci dummy PIL obrazek (grayscale 10x10)
        # Barva (intenzita) obrazku zavisi na planar_type: pokud 'ant_pw', barva 50, jinak 100
        val = 50 if planar_type == "ant_pw" else 100
        img = Image.new("L", (10, 10), color=val)
        return img


@pytest.fixture
def setup_roi_drawer():
    # Fixture pro pripravu testovaciho prostredi pro ROI_drawer_manual
    # Vytvori dummy dicom objekt (instanci DummyDicomObj)
    dummy_dicom = DummyDicomObj()
    # Vytvori slovnik dicom_obj, kde klice 1 a 2 odkazujou na dummy_dicom
    dicom_obj = {
        1: dummy_dicom,
        2: dummy_dicom,  # muze byt i jiny DummyDicomObj, ale zde je to stejny objekt
    }
    # Nastavi planar_type na 'ant_pw'
    planar_type = "ant_pw"
    # Vytvori slovnik img_labels, kde klic 2 obsahuje MagicMock objekt, ktery ma metodu config
    img_labels = {2: MagicMock()}
    # Nastavi velikost obrazku na (10, 10)
    size_image = (10, 10)

    # Vytvori instanci ROI_drawer_manual s pripravenymi parametry
    roi_drawer = ROI_drawer_manual(dicom_obj, planar_type, img_labels, size_image)
    # Vrati instanci pro dalsi testy
    return roi_drawer


def test_initialization(setup_roi_drawer):
    # Test inicializace ROI_drawer_manual pres fixture setup_roi_drawer
    roi = setup_roi_drawer
    # Overi, ze planar_type byl spravne nastaven na 'ant_pw'
    assert roi.planar_type == "ant_pw"
    # Overi, ze atribut image ma tvar (10,10)
    assert roi.image.shape == (10, 10)
    # Overi, ze seznam bodu ROI je na zacatku prazdny
    assert roi.roi_points == []
    # Overi, ze maska ROI je na zacatku None (neni definovana)
    assert roi.mask is None


def test_create_mask_empty_and_valid_polygon(setup_roi_drawer):
    # Test metody create_mask pri prazdnem a validnim polygonu
    roi = setup_roi_drawer

    # Test prazdneho polygonu - kdyz neni zadny bod ROI
    roi.roi_points = []
    roi.create_mask()
    # Maska by mela mit stejny tvar jako obrazek
    assert roi.mask.shape == roi.image.shape
    # Maska by mela byt cela False (zadne pixely nejsou oznaceny)
    assert not roi.mask.any()

    # Test polygonu se 3 body (trojuhelnik umisteny uprostred obrazku)
    roi.roi_points = [(2, 2), (7, 2), (4, 7)]
    roi.create_mask()
    # Maska by mela mit stejny tvar jako obrazek
    assert roi.mask.shape == roi.image.shape
    # V masce by melo byt nekolik pixelu oznaceno True (ROI vyplnena)
    assert roi.mask.sum() > 0


def test_remove_old_artists(setup_roi_drawer):
    # Test metody remove_old_artists, ktera odstranuje graficke objekty (artists)
    roi = setup_roi_drawer

    # Definice dummy tridy DummyArtist s metodou remove, ktera nastavi flag removed na True
    class DummyArtist:
        def remove(self):
            self.removed = True

    # Dummy trida DummyContour, ktera obsahuje seznam collections s DummyArtist objekty
    class DummyContour:
        def __init__(self):
            self.collections = [DummyArtist(), DummyArtist()]

    # Nastavi do ROI atribut contour na instanci DummyContour
    roi.contour = DummyContour()
    # Nastavi polygon_patch na DummyArtist instanci
    roi.polygon_patch = DummyArtist()
    # Nastavi point_patches na seznam dvou DummyArtist instanci
    roi.point_patches = [DummyArtist(), DummyArtist()]

    # Zavola testovanou metodu, ktera by mela odstranit vsechny tyto artists
    roi.remove_old_artists()

    # Ocekavame, ze po odstraneni bude atribut contour None
    assert roi.contour is None
    # Ocekavame, ze polygon_patch bude None
    assert roi.polygon_patch is None
    # Ocekavame, ze seznam point_patches bude prazdny
    assert roi.point_patches == []


def test_on_select_calls_mask_and_display(setup_roi_drawer, monkeypatch):
    # Test, zda metoda on_select nastavi spravne body ROI a zavola potrebne metody
    roi = setup_roi_drawer

    # Nahradime metody create_mask, display_results a apply_roi_to_all_images MagicMock objekty
    roi.create_mask = MagicMock()
    roi.display_results = MagicMock()
    roi.apply_roi_to_all_images = MagicMock()

    # Definujeme seznam bodu polygonu (vertices)
    verts = [(1, 1), (5, 1), (3, 4)]
    # Zavolame metodu on_select s definovanym polygonem
    roi.on_select(verts)

    # Overime, ze atribut roi_points byl spravne nastaven na verts
    assert roi.roi_points == verts
    # Overime, ze metoda create_mask byla zavolana presne jednou
    roi.create_mask.assert_called_once()
    # Overime, ze display_results byla zavolana jednou
    roi.display_results.assert_called_once()
    # Overime, ze apply_roi_to_all_images byla zavolana jednou s parametrem planar_type
    roi.apply_roi_to_all_images.assert_called_once_with(roi.planar_type)


def test_apply_roi_to_all_images_basic(monkeypatch):
    # Testovani funkce apply_roi_to_all_images s realnym Tkinter root oknem (ale skrytym)
    root = tk.Tk()
    root.withdraw()  # okno nebude viditelne

    # Pripravime dummy dicom objekty ve slovniku
    dummy_dicom = DummyDicomObj()
    dicom_obj = {1: dummy_dicom, 2: dummy_dicom}
    # Pripravime mock objekty pro img_labels, ktere maji metodu config
    img_label_mock_1 = MagicMock()
    img_label_mock_2 = MagicMock()
    img_labels = {1: img_label_mock_1, 2: img_label_mock_2}

    # Vytvorime instanci ROI_drawer_manual s daty a velikosti obrazu
    roi = ROI_drawer_manual(dicom_obj, "ant_pw", img_labels, (10, 10))
    # Nastavime body ROI na ctverec (0,0),(0,5),(5,5),(5,0)
    roi.roi_points = [(0, 0), (0, 5), (5, 5), (5, 0)]

    # Zavolame metodu apply_roi_to_all_images pro planar_type 'ant_pw'
    roi.apply_roi_to_all_images("ant_pw")

    # Overime, ze metody config na img_label mock objektech byly zavolany (tedy obrazek byl aktualizovan)
    img_label_mock_1.config.assert_called()
    img_label_mock_2.config.assert_called()
    # Overime, ze atribut ant_roi na dummy dicom objektu je pole numpy nebo None (tedy byl nastaven)
    assert isinstance(dicom_obj[1].ant_roi, (np.ndarray, type(None)))

    # Uzavreme Tkinter okno, aby nenarustalo
    root.destroy()


def test_show_pixel_value_inside_and_outside(setup_roi_drawer):
    # Test metody show_pixel_value, ktera zobrazuje hodnotu pixelu pod kurzorem
    roi = setup_roi_drawer
    # Nastavime polygon ROI a vytvorime masku
    roi.roi_points = [(0, 0), (0, 5), (5, 5), (5, 0)]
    roi.create_mask()

    # Definujeme dummy tridu Event, ktera simuluje kurzorove udalosti
    class Event:
        def __init__(self, xdata, ydata, inaxes):
            self.xdata = xdata  # x pozice kurzoru ve souradnicich grafu
            self.ydata = ydata  # y pozice kurzoru
            self.inaxes = inaxes  # reference na axes (plochu) matplotlibu nebo None

    # Vytvorime udalost kurzoru uvnitr axes (obrazku)
    event_inside = Event(1, 1, roi.ax)
    # Zavolame metodu show_pixel_value s touto udalosti
    roi.show_pixel_value(event_inside)
    # Ocekavame, ze textovy prvek obsahuje retezec "Pixel Value" (tedy zobrazena hodnota pixelu)
    assert "Pixel Value" in roi.text.get_text()

    # Vytvorime udalost kurzoru mimo axes (inaxes == None)
    event_outside = Event(1, 1, None)
    # Vymazeme text
    roi.text.set_text("")
    # Zavolame show_pixel_value s udalosti mimo obrazek
    roi.show_pixel_value(event_outside)
    # Ocekavame, ze text zustava prazdny (zadna hodnota se nezobrazi)
    assert roi.text.get_text() == ""

    # Vytvorime udalost kurzoru s negativnimi souradnicemi (mimo obrazek)
    event_oob = Event(-1, -1, roi.ax)
    # Vymazeme text
    roi.text.set_text("")
    # Zavolame show_pixel_value s touto udalosti
    roi.show_pixel_value(event_oob)
    # Ocekavame, ze text zustava prazdny, protoze kurzor je mimo obrazek
    assert roi.text.get_text() == ""


#### PREMENOVY ZAKON --------------------------


def test_premenovy_zakon_basic_decay():
    # Test základniho pripadu premenoveho zakona - aktivita se po jedne polocasu snizi na polovinu
    aktivita = 1000.0  # pocatecni aktivita (napr. v Bq)
    reference_time = datetime(2025, 6, 1)  # referencni cas (zacatek mereni)
    nynejsi_time = reference_time + timedelta(
        days=8.02
    )  # cas o jednu polocasovou dobu pozdeji

    # Vypocet korigovane aktivity v danem case pomoci funkce premenovy_zakon
    corrected = premenovy_zakon(aktivita, reference_time, nynejsi_time)
    # Ocekavame, ze po jedne polocasove dobe je aktivita polovicni (s toleranci 1e-2)
    assert np.isclose(corrected, aktivita / 2, atol=1e-2)


def test_premenovy_zakon_no_time_difference():
    # Test kdy casovy rozdil mezi referencnim a aktualnim casem je nulovy
    aktivita = 500.0
    reference_time = datetime.now()  # aktualni cas
    nynejsi_time = reference_time  # stejny cas jako referencni

    # Funkce by mela vratit nezmenenou hodnotu aktivity, protoze cas se nezmenil
    corrected = premenovy_zakon(aktivita, reference_time, nynejsi_time)
    assert np.isclose(corrected, aktivita)


def test_premenovy_zakon_negative_time_difference():
    # Test kdy je "aktualni" cas driv nez referencni cas (negativni casovy rozdil)
    aktivita = 1000.0
    reference_time = datetime(2025, 6, 10)  # pozdejsi datum
    nynejsi_time = datetime(2025, 6, 1)  # drivejsi datum nez reference_time

    # Ocekavame, ze aktivita se zvysi, protoze casovy rozdil je zaporny (jakoby "v minulosti")
    corrected = premenovy_zakon(aktivita, reference_time, nynejsi_time)
    assert corrected > aktivita


def test_premenovy_zakon_raises_exception_on_invalid_input():
    # Test, ze funkce vyhodi vyjimku, pokud jsou vstupy neplatne (napr. neciselna aktivita)
    with pytest.raises(Exception):
        premenovy_zakon("neplatna_aktivita", datetime.now(), datetime.now())


#### TEW KOREKCE -------------------


def test_tew_correction_basic():
    # Test zakladniho pripadu TEW korekce
    # Vstupni obraz em a dve scatter mapy sc1 a sc2 jsou konstantni pole o stejne velikosti
    em = np.array([[100, 100], [100, 100]], dtype=float)
    sc1 = np.array([[10, 10], [10, 10]], dtype=float)
    sc2 = np.array([[10, 10], [10, 10]], dtype=float)

    # Volani funkce tew_correction, ktera vraci korektni obraz a odhad neurcitosti
    corrected_img, corrected_uncertainty = tew_correction(em, sc1, sc2)

    # Kontrola, ze vystupy maji stejny tvar jako vstupy
    assert corrected_img.shape == em.shape
    assert corrected_uncertainty.shape == em.shape

    # Vypocet ocekavane korekce podle formulace TEW (priblizny odhad scatteru a korekce)
    scatter_estimate_expected = (sc1 / 0.06 + sc2 / 0.06) * (0.2 / 2)
    corrected_expected = np.clip(
        em - scatter_estimate_expected, 0, None
    )  # hodnoty nesmi byt zaporne

    # Ocekavany vysledek musi byt blizky tomu, co vraci funkce
    assert np.allclose(corrected_img, corrected_expected)


def test_tew_correction_no_negative_values():
    # Test, ze korekovany obraz neobsahuje zaporne hodnoty i kdyz scatter mapy jsou velmi vysoke
    em = np.array([[1, 2], [3, 4]], dtype=float)
    sc1 = np.array([[100, 100], [100, 100]], dtype=float)
    sc2 = np.array([[100, 100], [100, 100]], dtype=float)

    corrected_img, _ = tew_correction(em, sc1, sc2)
    # Vsechny hodnoty v korekovanem obraze by mely byt >= 0 (clipping)
    assert np.all(corrected_img >= 0)


def test_tew_correction_output_types_and_shapes():
    # Test, ze vystupy funkce jsou typu numpy ndarray a maji spravny rozmer
    em = np.ones((3, 3)) * 10
    sc1 = np.ones((3, 3)) * 5
    sc2 = np.ones((3, 3)) * 5

    corrected_img, corrected_uncertainty = tew_correction(em, sc1, sc2)

    # Kontrola, ze vystupy jsou opravdu numpy pole
    assert isinstance(corrected_img, np.ndarray)
    assert isinstance(corrected_uncertainty, np.ndarray)
    # Kontrola spravnych rozmeru vystupu
    assert corrected_img.shape == (3, 3)
    assert corrected_uncertainty.shape == (3, 3)


def test_tew_correction_raises_exception_on_invalid_input():
    # Test, ze funkce vyhodi vyjimku pokud jsou vstupy nevalidni (napr. retezce misto poli)
    with pytest.raises(Exception):
        tew_correction("em_image", "sc1_image", "sc2_image")


#### TIME DIFERENCE ---------------------


def test_compute_time_differences_basic():
    # Test zakladni funkcionality: vypocet rozdilu casu v hodinach mezi referencni hodnotou a seznamem dat a casu
    reference = "01.06.2025 12:00"  # referencni datum a cas jako string ve formatu den.mesic.rok hodina:minuta
    dates = ["20250601", "20250601", "20250601"]  # seznam dat ve formatu RRRRMMDD
    times = ["120000", "130000", "140000"]  # seznam casu ve formatu HHMMSS
    result = compute_time_differences(
        reference, dates, times
    )  # volani funkce, ktera vraci rozdily v hodinach
    expected = [
        0.0,
        1.0,
        2.0,
    ]  # ocekavany vysledek, rozdil v hodinach od referencniho casu (0, 1, 2 hodiny)
    assert result == expected  # overeni, ze vysledek odpovida ocekavani


def test_compute_time_differences_with_milliseconds():
    # Test spravneho ignorovani milisekund v casovem retezci
    reference = "01.06.2025 12:00"
    dates = ["20250601"]
    times = [
        "123045.123"
    ]  # cas obsahuje milisekundy (123), ktere by mely byt ignorovany
    result = compute_time_differences(reference, dates, times)
    # vytvoreni ocekavane hodnoty bez milisekund
    expected = [
        (
            datetime.strptime("20250601 123045", "%Y%m%d %H%M%S")
            - datetime.strptime(reference, "%d.%m.%Y %H:%M")
        ).total_seconds()
        / 3600
    ]
    assert result == expected  # overeni, ze vysledek odpovida ocekavani bez milisekund


def test_compute_time_differences_different_dates():
    # Test vypoctu rozdilu mezi daty z ruznych dnu, kdy cas zustava stejny
    reference = "01.06.2025 12:00"
    dates = [
        "20250531",
        "20250601",
        "20250602",
    ]  # den pred, den samotny a den po referencnim datu
    times = ["120000", "120000", "120000"]  # casy jsou stejne (12:00:00)
    result = compute_time_differences(reference, dates, times)
    expected = [
        -24.0,
        0.0,
        24.0,
    ]  # rozdily v hodinach: -24 (den zpatky), 0, +24 (den dopredu)
    assert result == expected


def test_compute_time_differences_invalid_date_format():
    # Test spravneho vyhazovani chyby pri spatnem formatu data
    reference = "01.06.2025 12:00"
    dates = [
        "2025-06-01"
    ]  # spatny format data (ocekava se RRRRMMDD, zde je s pomlckami)
    times = ["120000"]
    with pytest.raises(Exception) as excinfo:
        compute_time_differences(reference, dates, times)  # ocekavame vyjimku
    # Overeni, ze vyjimka obsahuje ocekavany text o chybe pri parsovani
    assert "Error parsing date/time" in str(excinfo.value)


def test_compute_time_differences_invalid_time_format():
    # Test spravneho vyhazovani chyby pri spatnem formatu casu
    reference = "01.06.2025 12:00"
    dates = ["20250601"]
    times = ["12:00:00"]  # spatny format casu (ocekava se HHMMSS bez dvojtecek)
    with pytest.raises(Exception) as excinfo:
        compute_time_differences(reference, dates, times)  # ocekavame vyjimku
    assert "Error parsing date/time" in str(excinfo.value)


def test_compute_time_differences_empty_lists():
    # Test chovani funkce pri prazdnych vstupech - ocekavame prazdny seznam jako vystup
    reference = "01.06.2025 12:00"
    dates = []
    times = []
    result = compute_time_differences(reference, dates, times)
    assert result == []


#### GRAF_1 ---------------------------------


def test_init_sets_locale_and_rcparams(monkeypatch):
    # Test, ktery overuje spravne nastaveni locale a matplotlib rcParams pri inicializaci tridy Graf_1
    g = Graf_1(
        fontsize=12, title="Title", xlabel="X", ylabel="Y", figsize=(5, 4), dpi=100
    )

    # Locale by melo byt nastaveno na 'de_DE', ale muze se lisit podle dostupnosti locale v systemu
    try:
        current_locale = locale.setlocale(
            locale.LC_NUMERIC
        )  # zjisteni aktualniho nastaveni locale pro cisla
        assert (
            "de_DE" in current_locale
        )  # kontrola, jestli je v nastaveni locale 'de_DE'
    except locale.Error:
        # Pokud locale 'de_DE' neni dostupne, test se preskoci (aby nepadal na systemech bez teto locale)
        pytest.skip("Locale 'de_DE' neni dostupne na tomto systemu")

    # Kontrola, ze jsou spravne nastaveny parametry matplotlib pro formatovani a velikosti fontu
    assert (
        plt.rcParams["axes.formatter.use_locale"] is True
    )  # ověřuje, ze matplotlib pouziva locale pro formatovani cisel
    assert (
        plt.rcParams["font.size"] == 12
    )  # velikost fontu v cele grafice odpovida zadane hodnote
    assert (
        plt.rcParams["xtick.labelsize"] == 12
    )  # velikost popisku na ose x odpovida fontsize
    assert (
        plt.rcParams["ytick.labelsize"] == 12
    )  # velikost popisku na ose y odpovida fontsize

    # Kontrola, ze atributy instance Graf_1 byly spravne nastaveny pri inicializaci
    assert g.title == "Title"
    assert g.xlabel == "X"
    assert g.ylabel == "Y"
    assert g.legend_fontsize == 10  # defaultni velikost legendy je fontsize - 2


def test_init_legend_fontsize_custom():
    # Test nastaveni vlastni velikosti legendy pri vytvareni instance Graf_1
    g = Graf_1(
        fontsize=14,
        title="T",
        xlabel="X",
        ylabel="Y",
        figsize=(5, 4),
        dpi=100,
        legend_fontsize=8,
    )
    assert (
        g.legend_fontsize == 8
    )  # overi, ze nastaveny parametr legend_fontsize je spravne ulozen v instance


def test_plot_calls_matplotlib_plot(monkeypatch):
    # Test, ze metoda plot tridy Graf_1 vola spravne metodu plot z matplotlib a predava spravne argumenty
    g = Graf_1(fontsize=10, title="", xlabel="", ylabel="", figsize=(5, 4), dpi=100)
    called = {}  # slovnik na zaznamenani argumentu, ktere fake funkce prijme

    def fake_plot(
        x, y, marker, label=None, color=None, markeredgewidth=None, markersize=None
    ):
        # Tato funkce simulujici matplotlib.plot ulozi vsechny parametry do 'called' pro naslednou kontrolu
        called["x"] = x
        called["y"] = y
        called["marker"] = marker
        called["label"] = label
        called["color"] = color
        called["markeredgewidth"] = markeredgewidth
        called["markersize"] = markersize
        return None

    def fake_legend(loc=None, edgecolor=None, fontsize=None):
        # Simulovana funkce legend, ktera ulozi volane parametry
        called["legend"] = (loc, edgecolor, fontsize)
        return None

    # Pomoci monkeypatch nahradime skutecne metody plot a legend nasimi fake funkcemi
    monkeypatch.setattr(g.fig, "plot", fake_plot)
    monkeypatch.setattr(g.fig, "legend", fake_legend)

    # Volame testovanou metodu plot tridy Graf_1
    g.plot(
        [1, 2, 3],
        [4, 5, 6],
        marker="o",
        label_data="lbl",
        color="r",
        markeredgewidth=1,
        markersize=5,
    )

    # Overeni, ze vsechny argumenty byly spravne predany a ulozeny ve fake plot a legend metodach
    assert called["x"] == [1, 2, 3]
    assert called["y"] == [4, 5, 6]
    assert called["marker"] == "o"
    assert called["label"] == "lbl"
    assert called["color"] == "r"
    assert called["markeredgewidth"] == 1
    assert called["markersize"] == 5
    assert called["legend"] == ("best", "black", g.legend_fontsize)


def test_errorbar_calls_matplotlib_errorbar(monkeypatch):
    # Test, ze metoda errorbar tridy Graf_1 vola spravne matplotlib.errorbar s ocekavanymi argumenty
    g = Graf_1(fontsize=10, title="", xlabel="", ylabel="", figsize=(5, 4), dpi=100)
    called = {}

    def fake_errorbar(
        x,
        y,
        yerr,
        fmt,
        marker,
        markersize,
        ecolor,
        zorder,
        elinewidth,
        label,
        capsize,
        color,
    ):
        # Simulovana funkce errorbar, ktera ulozi vsechny prijate argumenty pro kontrolu
        called.update(locals())
        return None

    def fake_legend(loc=None, edgecolor=None, fontsize=None):
        called["legend"] = (loc, edgecolor, fontsize)
        return None

    # Nahrazeni metod matplotlib nasimi fake funkcemi
    monkeypatch.setattr(g.fig, "errorbar", fake_errorbar)
    monkeypatch.setattr(g.fig, "legend", fake_legend)

    # Volani errorbar metody s argumenty pro test
    g.errorbar(
        x=[1, 2, 3],
        y=[4, 5, 6],
        yerr=[0.1, 0.2, 0.1],
        marker="o",
        label_data="lbl",
        color="r",
        markersize=5,
        sirka_nejistot=2,
        sirka_primky_nejistot=1,
    )

    # Overeni, ze vsechny parametry byly spravne predany do matplotlib.errorbar
    assert called["x"] == [1, 2, 3]
    assert called["y"] == [4, 5, 6]
    assert called["yerr"] == [0.1, 0.2, 0.1]
    assert called["fmt"] == " "  # volani s prazdnym formatem pro errorbar
    assert called["marker"] == "o"
    assert called["markersize"] == 5
    assert called["ecolor"] == "black"
    assert called["zorder"] == 2
    assert called["elinewidth"] == 1
    assert called["label"] == "lbl"
    assert called["capsize"] == 2
    assert called["color"] == "r"
    assert called["legend"] == ("best", "black", g.legend_fontsize)


def test_plot_and_errorbar_legend_fontsize(monkeypatch):
    # Test, ktery overuje, ze velikost pismen v legende (legend_fontsize) je spravne pouzita v obou metodech plot a errorbar
    g = Graf_1(
        fontsize=10,
        title="",
        xlabel="",
        ylabel="",
        figsize=(5, 4),
        dpi=100,
        legend_fontsize=7,
    )
    monkeypatch.setattr(
        g.fig, "plot", lambda *args, **kwargs: None
    )  # nahrazeni metod prazdnymi lambda funkci
    monkeypatch.setattr(g.fig, "errorbar", lambda *args, **kwargs: None)

    legend_called = []  # seznam, kam budeme ukládat volane velikosti legendy

    def fake_legend(loc=None, edgecolor=None, fontsize=None):
        legend_called.append(
            fontsize
        )  # pri kazdem volani ulozi velikost pismen do seznamu
        return None

    monkeypatch.setattr(
        g.fig, "legend", fake_legend
    )  # nahrazeni matplotlib legend funkce

    # Zavolame metody plot a errorbar, aby vyvolaly volani legendy
    g.plot([1], [2], "o", "lbl", "r", 1, 5)
    g.errorbar([1], [2], [0.1], "o", "lbl", "r", 5, 2, 1)

    # Overime, ze vsechny volani legendy pouzivaji stejnou velikost fontu, kterou jsme nastavili (7)
    assert all(fontsize == 7 for fontsize in legend_called)


#### FITOVÁNÍ ---------------


def test_riu_uptace_fce_basic():
    # Zakladni test funkce riu_uptace_fce
    # Definujeme x - pole hodnot (casove body)
    x = np.array([0, 1, 2])
    # Nastavime konstanty pro parametry
    k_t = 0.1
    k_B = 0.2
    k_T = 0.05

    # Vypocitame ocekavany vysledek podle vzorce
    expected = (k_t / (k_B - k_T)) * (np.exp(-k_T * x) - np.exp(-k_B * x))
    # Zavolame testovanou funkci s parametry
    result = riu_uptace_fce(x, k_t, k_B, k_T)

    # Overime, ze vypoctene hodnoty se shoduji s ocekavanymi (relativni tolerance 1e-7)
    np.testing.assert_allclose(result, expected, rtol=1e-7)


def test_basic_fit():
    # Test fitovani funkce riu_fit na idealni data bez sumu
    # Vytvorime pole x rovnomerne rozlozene od 0 do 10, 50 bodu
    x = np.linspace(0, 10, 50)
    # Definujeme "pravdive" parametry
    k_t_true, k_B_true, k_T_true = 0.05, 0.1, 0.005
    # Vypocteme y hodnoty podle funkce (idealni data)
    y = (k_t_true / (k_B_true - k_T_true)) * (
        np.exp(-k_T_true * x) - np.exp(-k_B_true * x)
    )
    # Zavolame fitovaci funkci
    params, errors, covar = riu_fit((x, y))
    # Overime, ze vsechny parametry jsou konecne (neni nan, inf)
    assert np.all(np.isfinite(params))
    # Overime, ze odhady chyb jsou nezaporne
    assert np.all(errors >= 0)
    # Overime, ze kovariancni matice ma spravny rozmer (3x3)
    assert covar.shape == (3, 3)


def test_fit_with_noise():
    # Test fitovani na datech s pridaným normalnim sumem
    x = np.linspace(0, 10, 50)
    k_t_true, k_B_true, k_T_true = 0.05, 0.1, 0.005
    # Idealni y
    y = (k_t_true / (k_B_true - k_T_true)) * (
        np.exp(-k_T_true * x) - np.exp(-k_B_true * x)
    )
    # Pridani sumu s rozptylem 0.01
    y_noisy = y + np.random.normal(0, 0.01, size=y.shape)
    # Fitujeme na sumem znecistenych datech
    params, errors, covar = riu_fit((x, y_noisy))
    # Overime, ze vysledne parametry jsou konecne
    assert np.all(np.isfinite(params))


def test_fit_with_y_err():
    # Test fitovani s vahami chyb (y_err)
    x = np.linspace(0, 10, 50)
    k_t_true, k_B_true, k_T_true = 0.05, 0.1, 0.005
    y = (k_t_true / (k_B_true - k_T_true)) * (
        np.exp(-k_T_true * x) - np.exp(-k_B_true * x)
    )
    y_noisy = y + np.random.normal(0, 0.01, size=y.shape)
    # Vytvorime konstantni chyby y_err
    y_err = np.full_like(y_noisy, 0.01)
    # Fit s vahami chyb
    params, errors, covar = riu_fit((x, y_noisy), y_err=y_err)
    # Overime, ze parametry jsou konecne
    assert np.all(np.isfinite(params))


def test_fit_with_zero_error():
    # Test, ze pokud jsou chyby y_err nulove, fit vyhodi vyjimku
    x = np.linspace(0, 10, 10)
    y = np.ones_like(x)
    y_err = np.zeros_like(y)
    with pytest.raises(Exception):
        riu_fit((x, y), y_err=y_err)


def test_empty_input():
    # Test, ze prazdny vstup vyhodi vyjimku
    with pytest.raises(Exception):
        riu_fit(([], []))


def test_mismatched_lengths():
    # Test, ze ruzna delka vstupnich poli x a y vyhodi vyjimku
    with pytest.raises(Exception):
        riu_fit(([1, 2, 3], [1, 2]))


def test_nan_in_data():
    # Test, ze pokud jsou v datech NaN hodnoty, fit vyhodi vyjimku
    x = np.array([0, 1, 2])
    y = np.array([1, np.nan, 2])
    with pytest.raises(Exception):
        riu_fit((x, y))


def test_parameters_stderr_not_none():
    # Test, ze chyby odhadu parametru (stderr) nejsou None
    x = np.linspace(0, 10, 20)
    y = (0.05 / (0.1 - 0.005)) * (np.exp(-0.005 * x) - np.exp(-0.1 * x))
    params, errors, covar = riu_fit((x, y))
    # Overime, ze vsechny chyby jsou definovane (neni None)
    assert all(e is not None for e in errors)


def test_covar_shape():
    # Test tvaru kovariancni matice, musi byt 3x3 (3 parametry)
    x = np.linspace(0, 10, 20)
    y = (0.05 / (0.1 - 0.005)) * (np.exp(-0.005 * x) - np.exp(-0.1 * x))
    _, _, covar = riu_fit((x, y))
    assert covar.shape == (3, 3)
