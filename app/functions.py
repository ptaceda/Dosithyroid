import pydicom
import numpy as np
from PIL import Image, ImageTk, ImageDraw
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path
from matplotlib.patches import Circle, Polygon
import matplotlib.pyplot as plt
import locale
from datetime import datetime
from lmfit import Model


class dicom_image:
    def __init__(self):
        # Konstruktor tridy - inicializuje vsechny atributy na None.
        # Tyto atributy budou pozdeji obsahovat jednotlive planarni obrazky nebo metadata z DICOM souboru.

        # Planarni obrazky pro ruzne okna a projekce (anterior/posterior)
        self.ant_pw = (
            None  # Anteriorni obraz z hlavniho energetickeho okna (photopeak window)
        )
        self.pos_pw = None  # Posteriorni obraz z hlavniho energetickeho okna
        self.ant_lsw = None  # Anteriorni obraz z dolniho scatter okna
        self.pos_lsw = None  # Posteriorni obraz z dolniho scatter okna
        self.ant_usw = None  # Anteriorni obraz z horniho scatter okna
        self.pos_usw = None  # Posteriorni obraz z horniho scatter okna

        # Metadata k obrazum
        self.acq_date = None  # Datum akvizice
        self.acq_time = None  # Cas akvizice
        self.acq_dur = None  # Delka akvizice (pocet milisekund prevedeny na sekundy)

        # ROI (region of interest) - budou se pozdeji pouzivat pro zakresleni
        self.ant_roi = None
        self.pos_roi = None

    def load_dicom(self, dicom_path):
        """
        Nacte DICOM soubor z cesty `dicom_path`, extrahuje jednotlive planarni obrazy
        a ulozi je do atributu tridy. Zaroven nacte dulezita metadata.
        """
        try:
            # Nacteni DICOM souboru pomoci knihovny pydicom
            dicom_data = pydicom.dcmread(dicom_path)

            # Zkontroluj pocet obrazovych rovin (pole pixel_array)
            # Ocekavame, ze bude mit 6 rovin (ant/post pro PW, LSW, USW)
            if len(dicom_data.pixel_array) == 6:
                self.ant_pw = dicom_data.pixel_array[0]
                self.pos_pw = dicom_data.pixel_array[1]
                self.ant_lsw = dicom_data.pixel_array[2]
                self.pos_lsw = dicom_data.pixel_array[3]
                self.ant_usw = dicom_data.pixel_array[4]
                self.pos_usw = dicom_data.pixel_array[5]
            else:
                # Pokud jsou pouze 2 rovin (typicky jen PW), zbyvajici scatter okna nahradime poli jednicek
                # Tim zajistime kompatibilitu dalsiho zpracovani bez nutnosti dalsi validace
                self.ant_pw = dicom_data.pixel_array[0]
                self.pos_pw = dicom_data.pixel_array[1]
                self.ant_lsw = np.ones_like(self.ant_pw)
                self.pos_lsw = np.ones_like(self.pos_pw)
                self.ant_usw = np.ones_like(self.ant_pw)
                self.pos_usw = np.ones_like(self.pos_pw)

            # Nacteni zakladnich metadat z hlavicky DICOMu
            self.acq_date = dicom_data[0x0008, 0x0022].value  # Acquisition Date (DA)
            self.acq_time = dicom_data[0x0008, 0x0032].value  # Acquisition Time (TM)
            self.acq_dur = (
                dicom_data[0x0018, 0x1242].value * 0.001
            )  # Acquisition Duration (milisekundy na sekundy)

            # Urceni maximalnich hodnot v hlavnim okne (PW) – pro pozdejsi kontrastni normalizaci
            self.ant_max = np.max(self.ant_pw)
            self.pos_max = np.max(self.pos_pw)

            # Prevence deleni nulou – pokud je obraz prazdny, nastavime maximum na 1
            if self.ant_max == 0:
                self.ant_max = 1
            if self.pos_max == 0:
                self.pos_max = 1

            # Informacni vypisy do konzole
            print(f"Loaded DICOM file: {dicom_path}")
            print(f"Acquisition Date: {self.acq_date}")
            print(f"Acquisition Time: {self.acq_time}")
            print(f"Acquisition Duration: {self.acq_dur}")

        except Exception as e:
            # Chyba pri nacitani – vypiseme chybu a propagujeme dal
            print(f"Error loading DICOM file: {e}")
            raise Exception(f"Error loading DICOM file: {e}")

    def convert_to_image(self, planar_type="ant_pw"):
        """
        Prevede vybrany planarni obrazek na PIL image objekt.
        Pouziva se pro zobrazeni ve GUI nebo ulozeni do souboru.
        """
        try:
            # Vyber obrazoveho pole podle zadaneho typu
            if planar_type == "ant_pw":
                image_array = self.ant_pw
                # Normalizace na rozsah 0–255 pro zobrazeni (kontrastni transformace)
                image_array = image_array.astype(np.float32)
                image_array = np.clip((image_array / self.ant_max) * 255, 0, 255)
            elif planar_type == "pos_pw":
                image_array = self.pos_pw
                image_array = image_array.astype(np.float32)
                image_array = np.clip((image_array / self.pos_max) * 255, 0, 255)
            elif planar_type == "ant_lsw":
                image_array = self.ant_lsw
            elif planar_type == "pos_lsw":
                image_array = self.pos_lsw
            elif planar_type == "ant_usw":
                image_array = self.ant_usw
            elif planar_type == "pos_usw":
                image_array = self.pos_usw
            else:
                # Pokud zadany typ neni podporovan, vyhod vyjimku
                raise Exception(
                    f"Invalid planar type or image not available: {planar_type}"
                )

            # Prevod numpy pole na obrazek pomoci PIL – grayscale 8-bit
            image = Image.fromarray(image_array.astype(np.uint8))

            return image

        except Exception as e:
            # Osetreni chyby pri konverzi
            raise Exception(f"Error converting DICOM image to PIL: {e}")


def align_images(reference_image, moving_image, sigma=0.5):
    """
    Zarovna (zarovna registracne) dva obrazy pomoci konvoluce ve frekvencni domene (FFT),
    s volitelnym gaussovskym zhlazenim pred samotnym zarovnavanim.
    """

    # Kontrola, ze obrazky maji stejnou velikost
    if reference_image.shape != moving_image.shape:
        raise ValueError("Input images must have the same shape for alignment.")

    try:
        # --- 1. Gaussovske zhlazeni ---
        # Cilem je potlacit sum a jemne struktury, ktere by mohly zpusobit chybnou detekci maxima
        reference_image_smoothed = gaussian_filter(reference_image, sigma=sigma)
        moving_image_smoothed = gaussian_filter(moving_image, sigma=sigma)

        # --- 2. Vypocet konvoluce pres FFT ---
        # Pouziva se cross-korelacni metoda:
        # Posunem moving_image proti reference_image a hledanim pozice s nejvetsim prunikem obsahu
        # Reverzni indexace (::-1, ::-1) odpovida matematicke definici konvoluce
        convolution_result = fftconvolve(
            reference_image_smoothed, moving_image_smoothed[::-1, ::-1], mode="same"
        )

        # --- 3. Lokalizace nejvetsi hodnoty v konvolucnim poli ---
        # Pozice, kde moving_image nejlepe pasuje na reference_image
        y_max, x_max = np.unravel_index(
            np.argmax(convolution_result), convolution_result.shape
        )

        # --- 4. Vypocet realneho posunu ---
        # Stred obrazku je teoreticka "nulova" pozice
        # Posun je rozdil mezi pozici maxima a stredem
        shift_y = y_max - reference_image.shape[0] // 2
        shift_x = x_max - reference_image.shape[1] // 2

        # --- 5. Posun obrazu ---
        # Funkce np.roll provede cyklicky posun obrazku
        # (nevyplnuje nulami, pouze "zaroluje" hodnoty na druhy konec matice)
        aligned_moving_image = np.roll(
            moving_image, shift=(shift_y, shift_x), axis=(0, 1)
        )

        # Navrat posunuteho obrazku a hodnot posunu
        return aligned_moving_image, shift_x, shift_y

    except Exception as e:
        # Pokud se vyskytne chyba, vypiseme a znovu vyhodime vyjimku
        print(f"Error aligning images: {e}")
        raise Exception(f"Error aligning images: {e}")


def posunuti_image(image, shift_x, shift_y):
    try:
        # Posun obrazku o shift_x (vodorovne) a shift_y (svisle) pomoci funkce np.roll
        # shift=(shift_y, shift_x): prvni hodnota je posun po osach radku (vertikalne),
        # druha hodnota je posun po sloupcich (horizontalne)
        # axis=(0, 1): specifikuje, ze posun se tyka 2D matice [radky, sloupce]

        return np.roll(image, shift=(shift_y, shift_x), axis=(0, 1))

    except Exception as e:
        # Pokud se vyskytne chyba (napr. nespravny typ nebo rozmer vstupu), vypiseme ji
        print(f"Error shifting image: {e}")
        raise Exception(f"Error shifting image: {e}")


class ROI_drawer_manual:
    def __init__(self, dicom_obj, planar_type, img_labels, size_image):
        """
        Konstruktor tridy, ktera zajistuje kresleni a upravu ROI polygonu na obraze.

        Parametry:
        - dicom_obj: slovnik s DICOM objekty, kde klicem je index a hodnotou objekt obsahujici obraz
        - planar_type: retezec 'ant_pw' nebo 'pos_pw', urcujici, kterou projekci zobrazit a upravovat
        - img_labels: slovnik Tkinter Label widgetu, ktere slouzi k zobrazeni obrazku s ROI v GUI
        - size_image: cilova velikost zobrazeni obrazku v pixelech (napr. 256x256)

        V teto funkci se inicializuje graficke okno, obrazek, PolygonSelector pro kresleni polygonu,
        a dalsi pomocne promenne.
        """
        try:
            # Ulozeni vstupnich dat do atributu tridy
            self.dicom_obj = dicom_obj
            self.planar_type = planar_type
            self.img_labels = img_labels

            if isinstance(size_image, int):
                self.size_image = (size_image, size_image)
            else:
                self.size_image = size_image

            # Vyber obrazku z DICOM objektu podle planar_type (predni ci zadni projekce)
            # index 2 je pevne zvolen, muze byt problem pokud slovnik nema tento index
            self.image = (
                dicom_obj[2].ant_pw if planar_type == "ant_pw" else dicom_obj[2].pos_pw
            )

            # Vytvoreni matplotlib figure a axes pro vykreslovani
            self.fig, self.ax = plt.subplots(
                figsize=(13, 13)
            )  # velke platno pro pohodlne kresleni

            # Vykresleni obrazku do axes s sedou (gray) barevnou mapou
            self.ax.imshow(self.image, cmap="gray")

            # Inicializace prazdneho seznamu pro souradnice bodu ROI polygonu
            self.roi_points = []

            # Masku ROI nastavime na None, az se vybere polygon, bude vytvorena
            self.mask = None

            # Vytvoreni PolygonSelector widgetu, ktery umoznuje interaktivne kreslit polygon
            # Parametr useblit=True zlepsuje vykon pri prekreslovani
            self.selector = PolygonSelector(self.ax, self.on_select, useblit=True)

            # Nastaveni barvy a sirky polygonu kresleneho PolygonSelector - pro lepsi viditelnost
            bright_blue = (0.0, 1.0, 0.0)  # barva RGB, zelena (lime)
            for artist in self.selector.artists:
                artist.set_color(bright_blue)  # nastaveni barvy na jasne zelenou
                artist.set_linewidth(2)  # tloustka car 2 pixely

            # Pripojeni udalosti, ktera detekuje pohyb kurzoru mysi nad vykreslovacim oknem
            # Pri pohybu bude volana funkce show_pixel_value, ktera zobrazi hodnotu pixelu pod kurzorem
            self.fig.canvas.mpl_connect("motion_notify_event", self.show_pixel_value)

            # Vytvoreni textoveho labelu v rohu axes pro zobrazovani hodnoty pixelu pod kurzorem
            self.text = self.ax.text(
                0.05,
                0.95,
                "",  # pozice (x=5%, y=95% z axes souradnic)
                transform=self.ax.transAxes,  # souradnice v pomeru k axes (ne pixelove)
                color="yellow",
                fontsize=12,  # barva textu a velikost pismene
                bbox=dict(
                    facecolor="black", alpha=0.5
                ),  # pozadi textu s polopruzracnym cernym boxem
            )

            # Inicializace promennych, ktere budou slouzit k vykreslovani ROI
            self.contour = None  # cervena hranice ROI - bude ulozena contourova linka
            self.polygon_patch = None  # zeleny polygon kolem ROI (patch objekt)
            self.point_patches = []  # seznam patch objektu pro jednotlive body polygonu
            self.point_radius = 1  # polomer kruznic pro body polygonu, zvoleno male

        except Exception as e:
            # Pokud nastane chyba pri inicializaci, vypis ji a prehod vyjimku dale
            print(f"Error initializing ROI drawer: {e}")
            raise Exception(f"Error initializing ROI drawer: {e}")

    def show(self):
        """
        Metoda pro zobrazeni grafickeho okna s obrazkem a moznosti kresleni ROI.
        Zavola plt.show(), ktere blokuje dalsi beh programu dokud okno neni zavreno.
        """
        try:
            plt.show()
        except Exception as e:
            print(f"Error displaying ROI selection: {e}")
            raise Exception(f"Error displaying ROI selection: {e}")

    def on_select(self, verts):
        """
        Callback funkce, ktera je volana PolygonSelectorem po dokonceni (nebo uprave) polygonu ROI.
        Parameter verts je seznam bodu [(x,y), ...] definujicich polygon.

        Co se zde deje:
        - aktualizace seznamu bodu self.roi_points
        - vytvoreni binarni masky ROI na obrazku
        - prekresleni obrazku s novou ROI
        - aplikace ROI na vsechny obrazky v GUI a v dicom_obj
        """
        try:
            self.roi_points = verts  # ulozeni aktualnich souradnic polygonu
            self.create_mask()  # vytvoreni binarni masky (bool pole) pro ROI
            self.display_results()  # prekresleni grafiky v axes (polygon, hranice, body)
            self.apply_roi_to_all_images(
                self.planar_type
            )  # aktualizace vsech obrazku v GUI
        except Exception as e:
            print(f"Error processing ROI selection: {e}")
            raise Exception(f"Error processing ROI selection: {e}")

    def create_mask(self):
        """
        Vytvori binarni masku ROI ve forme 2D numpy pole shodne velikosti s obrazkem.
        Masku vytvori pomoci matplotlib.path.Path a funkce contains_points,
        ktera zjisti, ktere pixely patri do polygonu ROI.

        Postup:
        - z bodu polygonu vytvori Path objekt
        - vytvori grid souradnic vsech pixelu (x,y)
        - zjisti, ktere pixely jsou uvnitr polygonu
        - vysledek ve tvaru 2D pole True/False ulozi do self.mask
        """
        try:
            if not self.roi_points or len(self.roi_points) < 3:
                # Polygon musí mít minimálně 3 body, jinak nastavíme prázdnou masku
                self.mask = np.zeros_like(self.image, dtype=bool)
                return

            height, width = self.image.shape
            y, x = np.mgrid[:height, :width]
            points = np.vstack((x.ravel(), y.ravel())).T

            path = Path(self.roi_points)
            mask = path.contains_points(points)
            self.mask = mask.reshape(self.image.shape)

        except Exception as e:
            raise Exception(f"Error creating mask: {str(e)}")

    def remove_old_artists(self):
        """
        Pomocna metoda, ktera odstrani stare vykreslene objekty ROI z axes,
        aby nedochazelo k prekryvani nebo mnozstvi cary/bodu pri prekresleni.

        Odstrani:
        - cervene contour linie ROI
        - zeleny polygon patch
        - kruznice reprezentujici body polygonu
        """
        # Odstraneni contour linky pokud existuje
        if self.contour is not None:
            if hasattr(
                self.contour, "collections"
            ):  # contour muze mit vice kolekci (line segments)
                for coll in self.contour.collections:
                    coll.remove()
            self.contour = None

        # Odstraneni polygonu (zeleny ram) pokud existuje
        if self.polygon_patch is not None:
            self.polygon_patch.remove()
            self.polygon_patch = None

        # Odstraneni vsech bodu polygonu (kruznice)
        for p in self.point_patches:
            p.remove()
        self.point_patches.clear()  # vycisteni seznamu patch objektu

    def display_results(self):
        """
        Metoda pro prekresleni axes s aktualnim obrazkem a vykreslenim ROI polygonu a hranice.
        Postup:
        - vycistit axes (odstranit vse)
        - vykreslit obrazek (pozadi)
        - vykreslit cervenou hranici ROI (contour) podle binarni masky
        - vykreslit zeleny polygon kolem ROI
        - vykreslit jednotlive body polygonu jako male zelené kruznice
        - vytvorit a zobrazit text s pixelovou hodnotou (pocatecne prazdny)
        - provest redraw canvas pro aktualizaci obrazku
        """
        try:
            self.ax.cla()  # vycisteni axes od vsech grafickych prvku

            self.ax.imshow(self.image, cmap="gray")  # vykresleni podkladoveho obrazku

            # Pokud existuje maska, vykresli cervenou hranici ROI pomoci contour
            if self.mask is not None:
                self.contour = self.ax.contour(self.mask, colors="r", linewidths=2)
            else:
                self.contour = None

            # Pokud mame alespon dva body polygonu, vykresli zeleny polygon patch
            if len(self.roi_points) >= 2:
                self.polygon_patch = Polygon(
                    self.roi_points,
                    closed=True,
                    fill=False,
                    edgecolor="lime",
                    linewidth=2,
                )
                self.ax.add_patch(self.polygon_patch)

            # Vykresli jednotlive body polygonu jako male zelené kruznice
            self.point_patches.clear()
            for x, y in self.roi_points:
                circ = Circle(
                    (x, y), radius=self.point_radius, color="lime", picker=True
                )
                self.ax.add_patch(circ)
                self.point_patches.append(circ)

            # Vytvoreni textoveho prvku pro hodnotu pixelu (zatim prazdny)
            self.text = self.ax.text(
                0.05,
                0.95,
                "",
                transform=self.ax.transAxes,
                color="yellow",
                fontsize=12,
                bbox=dict(facecolor="black", alpha=0.5),
            )

            # Prekresleni canvas, aby se vse aktualizovalo
            self.fig.canvas.draw_idle()

        except Exception as e:
            print(f"Error displaying ROI contour: {e}")
            raise Exception(f"Error displaying ROI contour: {e}")

    def apply_roi_to_all_images(self, planar_type):
        """
        Metoda, ktera aplikuje aktualni ROI polygon na vsechny obrazky v slovniku dicom_obj.

        Vykresli cervenou linku kolem ROI do kazdeho obrazku (pomoci Pillow draw.line)
        a nasledne aktualizuje Tkinter Label widgety, ktere zobrazují obrazky v GUI.

        Dale nastavi do vsech dicom objektu binarni masku ROI pod atributy ant_roi nebo pos_roi
        podle planar_type.

        Pozor: metoda predpoklada, ze kazdy objekt v dicom_obj ma metodu convert_to_image a
        ze img_labels obsahuje odpovidajici Label widgety.
        """
        try:
            for key in self.dicom_obj.keys():
                # Prevod obrazku na RGB a vytvoreni kresliciho objektu Pillow
                image = (
                    self.dicom_obj[key].convert_to_image(self.planar_type)
                ).convert("RGB")
                draw = ImageDraw.Draw(image)

                # Vytvoreni seznamu bodu jako int tuple pro kresleni polygonu (zakulaceni souradnic)
                roi_points_int = [(int(x), int(y)) for x, y in self.roi_points]

                # Vykresleni polygonu cervenou ciarou
                if len(roi_points_int) > 1:
                    # Přidání prvního bodu na konec seznamu pro uzavření polygonu
                    roi_points_int.append(roi_points_int[0])
                    draw.line(roi_points_int, fill="red", width=2)

                # Konverze obrazku zpatky na Tkinter PhotoImage
                tk_img = ImageTk.PhotoImage(
                    image.resize(self.size_image, Image.Resampling.LANCZOS)
                )

                # Aktualizace Tkinter Labelu obrazku v GUI
                self.img_labels[key].config(image=tk_img)
                self.img_labels[
                    key
                ].image = tk_img  # ukladame referenci, aby nedoslo k odstraneni GC

                # Nastaveni binarni masky do dicom objektu (podle planar_type)
                if planar_type == "ant_pw":
                    self.dicom_obj[key].ant_roi = self.mask
                else:
                    self.dicom_obj[key].pos_roi = self.mask

        except Exception as e:
            print(f"Error applying ROI to images: {e}")
            raise Exception(f"Error applying ROI to images: {e}")

    def show_pixel_value(self, event):
        """
        Event handler pro zobrazovani hodnoty pixelu pod kurzorem mysi.

        Pokud je kurzor uvnitr obrazu, zjisti hodnotu pixelu a prepise text v axes.
        Pokud kurzor neni v oblasti obrazku, text se skryje.

        Dulezite: event.xdata a event.ydata jsou souradnice v datech axes (float),
        je treba je prevest na int indexy pixelu.
        """
        try:
            if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
                return  # nevoláme set_text mimo osu nebo pokud není souřadnice

            x, y = int(event.xdata), int(event.ydata)

            if x < 0 or y < 0 or x >= self.image.shape[1] or y >= self.image.shape[0]:
                return  # mimo rozsah

            value = self.image[y, x]
            self.text.set_text(f"Pixel Value: {value:.2f}")
            plt.draw()

        except Exception as e:
            raise Exception(f"Error displaying pixel value: {str(e)}")


def premenovy_zakon(aktivita, reference_time, nynejsi_time):
    """
    Aplikuje korekci aktivity podle casoveho premenoveho zakona (fyzikalniho polocteni).
    Funkce pocita, jak se aktivita snizila exponencialnim zakonem od referencniho casu po aktualni cas.
    """

    try:
        # Vypocet rozdilu casu v dnech mezi aktualnim a referencnim casem
        delta_days = (nynejsi_time - reference_time).total_seconds() / (24 * 3600)

        # Polocteni radionuklidu v dnech (zde pevne dane 8.02 dnu)
        half_life = 8.02

        # Exponencialni rozklad aktivity podle vzorce A = A0 * exp(-ln(2)/T_half * t)
        return aktivita * np.exp(-np.log(2) / half_life * delta_days)

    except Exception as e:
        # V pripade chyby vypise zpravu a vyhodi vyjimku dale
        print(f"Error in decay correction: {e}")
        raise Exception(f"Error in decay correction: {e}")


def tew_correction(em_image, sc1_image, sc2_image):
    """
    Provede korekci rozptylu pomoci metody Triple Energy Window (TEW).

    Metoda TEW pouziva dve vedlejsi okna na odhad rozptylu,
    ktery je odecten od emisniho obrazu pro odstraneni vlivu rozptylu.
    """

    try:
        # Vypocet chyb (standardni odchylky) jako odmocnina poctu castic (Poissonova statistika)
        em_error = np.sqrt(em_image)
        sc1_error = np.sqrt(sc1_image)
        sc2_error = np.sqrt(sc2_image)

        # Odhad rozptylu jako prumer upravenych hodnot z dvou rozptylovych oken
        # Koeficienty 0.06 a 0.2 jsou experimentalne urcene vahy/metricke konstanty
        scatter_estimate = (sc1_image / 0.06 + sc2_image / 0.06) * (0.2 / 2)

        # Vypocet nejistoty odhadu rozptylu z nejistot jednotlivych rozptylovych oken
        scatter_uncertainty = 0.2 / (2 * 0.06) * np.sqrt(sc1_error**2 + sc2_error**2)

        # Korekce emisniho obrazu odectenim odhadu rozptylu,
        # soucasne orezani na minimum 0, aby nebyly zaporne hodnoty
        corrected_image = np.clip(em_image - scatter_estimate, a_min=0, a_max=None)

        # Nejistota korektniho obrazu spocitana jako suma chyb emisniho obrazu a rozptylu (nezavisle chyby)
        corrected_uncertainty = np.sqrt(em_error**2 + scatter_uncertainty**2)

        # Vrati korektni obraz a jeho nejistotu
        return corrected_image, corrected_uncertainty

    except Exception as e:
        # V pripade chyby vypise info a vyhodi vyjimku dale
        print(f"Error in TEW correction: {e}")
        raise Exception(f"Error in TEW correction: {e}")


def compute_time_differences(reference_date_time, dates, times):
    """
    Vypocita casove rozdily v hodinach vzhledem k referencnimu datu a casu.
    :return: Seznam casovych rozdilu v hodinach mezi kazdym datem/casem a referenci.
    """

    time_differences = []  # Inicializace seznamu pro vysledne casove rozdily

    # Prevod referencniho data a casu z retezce na datetime objekt
    reference = datetime.strptime(reference_date_time, "%d.%m.%Y %H:%M")

    try:
        # Projdeme kazdy par data a casu (z pole dates a times)
        for date_str, time_str in zip(dates, times):
            try:
                # Odstranime pripadne desetinne casti sekund z casu (napr. milisekundy)
                time_str = time_str.split(".")[0]  # Pouze cela sekunda

                # Spojeni data a casu do jednoho retezce
                dt_str = f"{date_str} {time_str}"

                # Prevod spojeneho retezce na datetime objekt
                # Format odpovida napriklad: '20230612 134501' -> 12.6.2023 13:45:01
                current_dt = datetime.strptime(dt_str, "%Y%m%d %H%M%S")

                # Vypocet rozdilu mezi aktualnim a referencnim casem v sekundach,
                # pak prevod na hodiny vydelenim 3600
                diff = (current_dt - reference).total_seconds() / 3600

                # Pridani vypocteneho rozdilu do seznamu
                time_differences.append(diff)

            except Exception as ve:
                # Pokud nastane chyba pri prevodu jednotlivych datumu/casu,
                # vypise chybovou zpravu a vyhodi vyjimku dale
                print(f"Error parsing date/time '{date_str} {time_str}': {ve}")
                raise Exception(
                    f"Error parsing date/time '{date_str} {time_str}': {ve}"
                )

    except Exception as e:
        # Chytani necekanych chyb pri cele funkci
        print(f"Unexpected error while computing time differences: {e}")
        raise Exception(f"Unexpected error while computing time differences: {e}")

    # Vraci seznam casovych rozdilu v hodinach
    return time_differences


class Graf_1:
    def __init__(
        self, fontsize, title, xlabel, ylabel, figsize, dpi, legend_fontsize=None
    ) -> None:
        # Nastavi lokalizaci cisel tak, aby se v grafech pouzivala carka misto tecky (napr. 1,23 misto 1.23)
        locale.setlocale(locale.LC_NUMERIC, "de_DE")

        # Reset nastaveni matplotlib na vychozi hodnoty
        plt.rcdefaults()

        # Povoli formatovani os podle lokalniho nastaveni (tedy s carkou misto tecky)
        plt.rcParams["axes.formatter.use_locale"] = True

        # Nastavi velikost fontu pro vsechny prvky grafu (osove popisky, cisla ticku apod.)
        plt.rcParams["font.size"] = fontsize
        plt.rcParams["xtick.labelsize"] = fontsize
        plt.rcParams["ytick.labelsize"] = fontsize

        # Vytvori figure a jeden subplot (os) s danymi rozmery (figsize) a dpi (rozliseni)
        self.Figure, self.fig = plt.subplots(figsize=figsize, dpi=dpi)

        # Ulozi si do instance zadany titulek grafu a popisky os
        self.title = title
        self.ylabel = ylabel
        self.xlabel = xlabel

        # Nastavi velikost fontu legendy; pokud neni zadana, pouzije se o neco mensi nez hlavni font
        self.legend_fontsize = (
            legend_fontsize if legend_fontsize is not None else fontsize - 2
        )

        # Nastavi popisky os a titulek na vytvorenem subplotu
        self.fig.set_xlabel(self.xlabel)
        self.fig.set_ylabel(self.ylabel)
        self.fig.set_title(self.title)

        # Zapne sit (mřízku) v grafu s cernou barvou, carkovanou carou a tenkym provednim
        self.fig.grid(color="black", ls="-.", lw=0.25)

    def plot(self, x, y, marker, label_data, color, markeredgewidth, markersize):
        # Vykresli data jako spojeny graf se znacky (marker) pro kazdy bod
        # x a y jsou souradnice, label_data je popisek pro legendu
        # color je barva car, markeredgewidth - sirka okraje markeru, markersize - velikost markeru
        self.fig.plot(
            x,
            y,
            marker,
            label=label_data,
            color=color,
            markeredgewidth=markeredgewidth,
            markersize=markersize,
        )

        # Zobrazi legendu s pozadovanym fontem a cernym okrajem
        self.fig.legend(loc="best", edgecolor="black", fontsize=self.legend_fontsize)

    def errorbar(
        self,
        x,
        y,
        yerr,
        marker,
        label_data,
        color,
        markersize,
        sirka_nejistot,
        sirka_primky_nejistot,
    ):
        # Vykresli body se znackami (marker) a s chybovymi useckami (error bars) podle yerr
        # sirka_nejistot je velikost carky na konci chybove usecky (capsize)
        # sirka_primky_nejistot je sirka samotne chybove usecky (elinewidth)
        # zorder=2 zarucuje, ze chybove usecky budou vykresleny nad dalsimi prvky grafu
        self.fig.errorbar(
            x=x,
            y=y,
            yerr=yerr,
            fmt=" ",
            marker=marker,
            markersize=markersize,
            ecolor="black",
            zorder=2,
            elinewidth=sirka_primky_nejistot,
            label=label_data,
            capsize=sirka_nejistot,
            color=color,
        )

        # Zobrazi legendu s pozadovanym fontem a cernym okrajem
        self.fig.legend(loc="best", edgecolor="black", fontsize=self.legend_fontsize)


def riu_uptace_fce(x, k_t, k_B, k_T):
    # Modelova funkce pro RIU (radioaktivni uptake)
    # x je cas (napr. v hodinach)
    # k_t, k_B, k_T jsou parametry modelu, ktere se budou fitovat
    # Vypocita hodnotu podle vzorce s exponencialami
    return (k_t / (k_B - k_T)) * (np.exp(-k_T * x) - np.exp(-k_B * x))


def riu_fit(x_a_y_data, y_err=None):
    try:
        # Rozdeli vstupni data na cas (x) a hodnoty RIU (y)
        cas_h = np.array(x_a_y_data[0])  # cas v hodinach
        riu_values = np.array(x_a_y_data[1])  # namerene hodnoty RIU

        # Vytvori model na zaklade funkce riu_uptace_fce
        model = Model(riu_uptace_fce)

        # Inicializuje parametry fitu s pocatecnimi odhady
        params = model.make_params(k_t=0.05, k_B=0.1, k_T=0.005)

        if y_err is not None:
            # Pokud jsou zadane nejistoty hodnot y, prevede je na numpy pole
            y_err = np.array(y_err)

            # Nulove hodnoty nejistot nahradi minimalni kladnou hodnotou, aby se vyhnulo deleni nulou
            y_err[y_err == 0] = np.min(y_err[y_err > 0])

            # Vypocita vahy pro fit jako 1/(nejistota^2) (vahy podle inverzni variance)
            weights = (1 / y_err) ** 2
        else:
            # Pokud nejsou zadane nejistoty, vahy nejsou pouzity
            weights = None

        # Fit provede v loopu 5 iteraci, cimz se snazi lepe konvergovat k optimu
        for i in range(5):
            result = model.fit(riu_values, params, x=cas_h, weights=weights)
            params = result.params  # aktualizuje parametry pro dalsi iteraci

        # Vytažení hodnot parametru z vysledku fitu
        k_t = result.params["k_t"].value
        k_B = result.params["k_B"].value
        k_T = result.params["k_T"].value

        # Vytažení standardnich odchylek (chyby) parametru z fitu
        k_t_err = result.params["k_t"].stderr
        k_B_err = result.params["k_B"].stderr
        k_T_err = result.params["k_T"].stderr

        # Vytažení kovarianční matice, ktera popisuje korelace mezi parametry
        covar = result.covar

        # Vrati hodnoty parametru, jejich chyby a kovariančni matici jako numpy pole
        return (
            np.array([k_t, k_B, k_T]),  # Hodnoty parametru
            np.array([k_t_err, k_B_err, k_T_err]),  # Chyby parametru
            covar,  # Kovariančni matice
        )

    except Exception as e:
        # Pri chybe vypise informaci a vyhodi vyjimku dale
        print(f"Error in riu_fit: {e}")
        raise Exception(f"Error in riu_fit: {e}")
