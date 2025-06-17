import os
from jinja2 import Environment, FileSystemLoader
from xhtml2pdf import pisa
from xhtml2pdf.default import DEFAULT_FONT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.fonts import addMapping


def register_fonts():
    font_dir = os.path.join(os.getcwd(), "fonts")
    pdfmetrics.registerFont(
        TTFont(
            "TeXGyreHeros", os.path.join(font_dir, "dehinted-TeXGyreHeros-Regular.ttf")
        )
    )
    pdfmetrics.registerFont(
        TTFont(
            "TeXGyreHeros-Bold",
            os.path.join(font_dir, "dehinted-TeXGyreHeros-Bold.ttf"),
        )
    )
    pdfmetrics.registerFont(
        TTFont(
            "TeXGyreHeros-Italic",
            os.path.join(font_dir, "dehinted-TeXGyreHeros-Italic.ttf"),
        )
    )
    pdfmetrics.registerFont(
        TTFont(
            "TeXGyreHeros-BoldItalic",
            os.path.join(font_dir, "dehinted-TeXGyreHeros-BoldItalic.ttf"),
        )
    )

    addMapping("TeXGyreHeros", 0, 0, "TeXGyreHeros")
    addMapping("TeXGyreHeros", 1, 0, "TeXGyreHeros-Bold")
    addMapping("TeXGyreHeros", 0, 1, "TeXGyreHeros-Italic")
    addMapping("TeXGyreHeros", 1, 1, "TeXGyreHeros-BoldItalic")


def render_pdf(template_path, output_path, context):
    env = Environment(loader=FileSystemLoader(os.path.dirname(template_path)))
    template = env.get_template(os.path.basename(template_path))
    html_content = template.render(context)

    with open(output_path, "wb") as f:
        pisa_status = pisa.CreatePDF(html_content, dest=f)
    return pisa_status.err


if __name__ == "__main__":
    register_fonts()

    # Použij svůj font jako defaultní sans-serif
    DEFAULT_FONT["sans-serif"] = "TeXGyreHeros"

    data = {
        "dozi_or_terap": "Posterapeutická dozimetrie",
        "pacient_jmeno": "Pan Tajný",
        "datum_narozeni": "00.00.0000",
        "diagnoza": "Toxický uzel v pravém laloku ŠŽ",
        "radiofarmakum": "kapsle Na[131I]I",
        "aktivita": "537 MBq",
        "datum_aktivita": "05.03.2025 07:20",
        "datum_aplikace": "12.02.2025 08:15",
        "aplikovana_aktivita": "535,9 MBq",
        "cilovy_objem": "PL ŠŽ 30x30x47mm ≃ 23,3 g dle UZ 20.02.2025, MUDr. Pan Neznámý",
        "zariadeni": "GE Optima NM/CT 640",
        "cf": "7,77 cps/MBq",
        "uptake_image_path": "uptake_z_planaru.png",
        "k_t": "0,0557",
        "k_B": "0,1609",
        "k_T": "0,0060",
        "tiac": "58,08 h | 2,42 dne",
        "f_proklad": "44,21 %",
        "t_eff": "4,84 dne",
        "e_prumerna": "2,805 Gy·g / MBq·d",
        "d": "156",
        "dose_comparison_image_path": "porovnani.jpg",
        "datum_exportu": "13.03.2025",
    }

    err = render_pdf("template_file.html", "output_protocol.pdf", data)
    if err:
        print("❌ Chyba při generování PDF.")
    else:
        print("✅ PDF úspěšně vytvořeno jako output_protocol.pdf")
