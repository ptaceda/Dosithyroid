import sys
import os
import pytest
import numpy as np
from unittest.mock import MagicMock

# Přidáme do sys.path nadřazený adresář aktuálního souboru, aby Python našel modul 'app'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import aplikace


@pytest.fixture
def app():
    # Fixture pro testy, vytvori instanci aplikace s init_gui nastavenym na False
    # Tato instance bude pouzita v dalsich testech jako parametr 'app'
    return aplikace(init_gui=False)


def test_validate_input_activity(app):
    # Testujeme funkci validate_input_activity z aplikace
    # Overujeme, ze prazdny retezec je validni
    assert app.validate_input_activity("") is True
    # Overujeme, ze ciste cisla jsou validni
    assert app.validate_input_activity("123") is True
    # Overujeme, ze cislo s jednou desetinnou teckou je validni
    assert app.validate_input_activity("12.3") is True
    # Overujeme, ze dve tecky ve vstupu nejsou validni
    assert app.validate_input_activity("12..3") is False
    # Overujeme, ze pismena ve vstupu nejsou validni
    assert app.validate_input_activity("12a3") is False
    # Overujeme, ze vice nez jedna desetinna tecka neni validni
    assert app.validate_input_activity("12.3.4") is False


def test_validate_date_input(app):
    # Testujeme funkci validate_date_input z aplikace
    # Overujeme, ze prazdny retezec je validni (mozna povoleno nevyplneni)
    assert app.validate_date_input("") is True
    # Overujeme spravny format data a casu den.mesic.rok hh:mm (plny rok)
    assert app.validate_date_input("12.05.2025 14:30") is True
    # Overujeme spravny format se zkracenym rokem (posledni dve cislice)
    assert app.validate_date_input("12.05.25 14:30") is True
    # Overujeme, ze format s neukoncenym casem (dvojtecka na konci) neni validni
    assert app.validate_date_input("12.05.2025 14:30:") is False
    # Overujeme, ze format s sekundami neni validni (podle definice validace)
    assert app.validate_date_input("12.05.2025 14:30:00") is False
    # Overujeme, ze cas s pomlckou misto dvojtecky neni validni
    assert app.validate_date_input("12.05.2025 14-30") is False
    # Overujeme, ze format data s lomitky misto tecek neni validni
    assert app.validate_date_input("12/05/2025") is False


#### Test DT korekce


class DummyDicomImage:
    def __init__(self, ant_pw, pos_pw, acq_dur):
        # Inicializace dummy DICOM obrazku s atributy:
        # ant_pw  - numpy pole s hodnotami ant (predni) power
        # pos_pw  - numpy pole s hodnotami pos (zadni) power
        # acq_dur - doba akvizice (zaznamu) v sekundach
        self.ant_pw = ant_pw
        self.pos_pw = pos_pw
        self.acq_dur = acq_dur


def test_DT_correction_from_main(tmp_path):
    # Vytvoreni instance aplikace bez GUI (parametr init_gui=False)
    app = aplikace(init_gui=False)

    # Nastaveni atributu, ze korekce MD jeste nebyla provedena
    app.provedeni_korekce_MD = False

    # Nastaveni vystupni slozky na docasny adresar pro testy
    app.output_folder = tmp_path

    # Vytvoreni fake DICOM obrazku jako slovnik, klic je index
    app.dicom_images = {
        0: DummyDicomImage(np.array([100, 200, 300]), np.array([150, 250, 350]), 10),
        1: DummyDicomImage(np.array([400, 500, 600]), np.array([450, 550, 650]), 20),
    }

    # Nastaveni parametru md_data, pravdepodobne konstant pro korekci
    app.md_data = {"ant_pw": 0.0001, "pos_pw": 0.0001}

    # Nastaveni safe_call na lambda, ktera jen zavola danou funkci (nahrazeni pro test)
    app.safe_call = lambda func, *args, **kwargs: func(*args, **kwargs)

    # Nahrazeni metody update_image_labels mock objektem (pro sledovani volani)
    app.update_image_labels = MagicMock()

    # Pridani dalsich potrebnych atributu jako mocky
    app.img_labels_ant = MagicMock()
    app.img_labels_pos = MagicMock()
    app.image_size = (100, 100)  # velikost obrazku (napr. pro vykreslovani)

    # Volani testovane metody DT_correction, ktera provede korekci na datech
    app.DT_correction()

    # Kontrola, ze po volani je priznak korekce nastaven na True (korekce probehla)
    assert app.provedeni_korekce_MD is True

    # Kontrola, ze soubor s parametry korekce byl vytvoren ve vystupni slozce
    assert (tmp_path / "DT_correction_params.txt").exists()

    # Kontrola, ze v souboru je text indikujici pouziti korekce pomoci Lambert W funkce
    assert (
        "Correction applied by Lambert W function"
        in (tmp_path / "DT_correction_params.txt").read_text()
    )

    # Kontrola, ze metoda update_image_labels byla opravdu zavolana (volana aspon jednou)
    assert app.update_image_labels.called
