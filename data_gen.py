"""
data_gen.py — Sample PDF Generator for AI Engine
Generates two sample PDF files in the data/ folder:
  1. sample_po.pdf   — Sample B2B Purchase Order
  2. sample_docs.pdf — Sample technical document (Safety Data Sheet)
"""

import os
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def base_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("DocTitle",   parent=styles["Title"],  fontSize=16, spaceAfter=4))
    styles.add(ParagraphStyle("SubTitle",   parent=styles["Normal"], fontSize=11, spaceAfter=2,
                               textColor=colors.HexColor("#555555")))
    styles.add(ParagraphStyle("SectionH",   parent=styles["Heading2"], fontSize=12,
                               spaceBefore=10, spaceAfter=4,
                               textColor=colors.HexColor("#1a3a5c")))
    styles.add(ParagraphStyle("BodyText2",  parent=styles["Normal"], fontSize=9,  leading=13))
    styles.add(ParagraphStyle("SmallLabel", parent=styles["Normal"], fontSize=8,
                               textColor=colors.HexColor("#777777")))
    styles.add(ParagraphStyle("TableHead",  parent=styles["Normal"], fontSize=9,
                               textColor=colors.white, alignment=TA_CENTER))
    styles.add(ParagraphStyle("TableCell",  parent=styles["Normal"], fontSize=9, leading=12))
    styles.add(ParagraphStyle("TableRight", parent=styles["Normal"], fontSize=9,
                               leading=12, alignment=TA_RIGHT))
    return styles


def hr(width="100%", thickness=0.5, color=colors.HexColor("#cccccc")):
    return HRFlowable(width=width, thickness=thickness, color=color, spaceAfter=4)


# ---------------------------------------------------------------------------
# 1. sample_po.pdf — Purchase Order B2B
# ---------------------------------------------------------------------------

def generate_sample_po(path: str):
    doc = SimpleDocTemplate(
        path, pagesize=A4,
        rightMargin=20 * mm, leftMargin=20 * mm,
        topMargin=18 * mm, bottomMargin=18 * mm,
    )
    styles = base_styles()
    story = []

    # ── Header ──────────────────────────────────────────────────────────────
    header_data = [[
        Paragraph("<b>PT SENTOSA AROMATICS</b><br/>"
                  "Jl. Industri Raya No. 45, Tangerang 15122<br/>"
                  "Telp: (021) 5588-1234 | Fax: (021) 5588-1235<br/>"
                  "Email: purchasing@sentosaaromatics.co.id", styles["BodyText2"]),
        Paragraph("<b>PURCHASE ORDER</b><br/>"
                  "<font size=9 color='#555555'>No: <b>PO-2026-00123</b></font><br/>"
                  "<font size=8 color='#777777'>Tanggal: 10 September 2026</font><br/>"
                  "<font size=8 color='#777777'>Tgl Pengiriman: 20 September 2026</font>",
                  ParagraphStyle("HDRight", parent=styles["Normal"],
                                 fontSize=10, alignment=TA_RIGHT))
    ]]
    header_tbl = Table(header_data, colWidths=[95 * mm, 80 * mm])
    header_tbl.setStyle(TableStyle([
        ("VALIGN",     (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(header_tbl)
    story.append(hr())
    story.append(Spacer(1, 4 * mm))

    # ── Vendor & Deliver-To ─────────────────────────────────────────────────
    addr_data = [[
        Paragraph("<b>Kepada Yth:</b><br/>"
                  "PT Multisari Indoprima<br/>"
                  "Jl. Pahlawan Revolusi No. 7, Jakarta Timur 13430<br/>"
                  "Telp: (021) 8610-9900<br/>"
                  "Attn: Bagian Penjualan", styles["BodyText2"]),
        Paragraph("<b>Kirimkan ke:</b><br/>"
                  "Gudang PT Sentosa Aromatics<br/>"
                  "Jl. Industri Raya No. 45, Tangerang 15122<br/>"
                  "Contact: Bpk. Hendra Wijaya<br/>"
                  "HP: 0812-3456-7890", styles["BodyText2"]),
    ]]
    addr_tbl = Table(addr_data, colWidths=[87 * mm, 88 * mm])
    addr_tbl.setStyle(TableStyle([
        ("BOX",        (0, 0), (0, 0), 0.5, colors.HexColor("#cccccc")),
        ("BOX",        (1, 0), (1, 0), 0.5, colors.HexColor("#cccccc")),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f7f9fc")),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
        ("VALIGN",     (0, 0), (-1, -1), "TOP"),
        ("INNERGRID",  (0, 0), (-1, -1), 0.3, colors.HexColor("#dddddd")),
    ]))
    story.append(addr_tbl)
    story.append(Spacer(1, 6 * mm))

    # ── Line Items Table ────────────────────────────────────────────────────
    story.append(Paragraph("Rincian Pesanan", styles["SectionH"]))

    col_heads = ["No", "Kode Barang", "Deskripsi", "Qty", "Satuan", "Harga Satuan (IDR)", "Total (IDR)"]
    items = [
        (1, "ARO-LV-001", "Lavender Essential Oil 100ml",          50,  "Botol",   85_000,   4_250_000),
        (2, "ARO-BG-002", "Bergamot Fragrance Oil 500ml",          30,  "Botol",  135_000,   4_050_000),
        (3, "ARO-SN-003", "Sandalwood Base Note Concentrate 1L",   20,  "Liter",  420_000,   8_400_000),
        (4, "ARO-RS-004", "Rose Absolute Extract 50ml",            15,  "Botol",  680_000,  10_200_000),
        (5, "ARO-CT-005", "Citrus Top Note Blend 250ml",           40,  "Botol",   95_000,   3_800_000),
        (6, "PKG-BO-010", "Botol Kaca Amber 100ml (Kosong)",      200,  "Pcs",     8_500,   1_700_000),
        (7, "PKG-LB-011", "Label Stiker Custom 5x5cm",            500,  "Lembar",    750,     375_000),
    ]

    def fmt(n): return f"{n:,.0f}"

    table_data = [col_heads]
    for it in items:
        table_data.append([
            str(it[0]),
            it[1],
            it[2],
            fmt(it[3]),
            it[4],
            fmt(it[5]),
            fmt(it[6]),
        ])

    subtotal = sum(i[6] for i in items)
    ppn      = int(subtotal * 0.11)
    total    = subtotal + ppn

    table_data.append(["", "", "", "", "", "Subtotal",  fmt(subtotal)])
    table_data.append(["", "", "", "", "", "PPN 11%",   fmt(ppn)])
    table_data.append(["", "", "", "", "", "TOTAL IDR", fmt(total)])

    col_w = [10*mm, 28*mm, 58*mm, 14*mm, 16*mm, 30*mm, 27*mm]
    item_tbl = Table(table_data, colWidths=col_w, repeatRows=1)
    item_tbl.setStyle(TableStyle([
        # Header row
        ("BACKGROUND",   (0, 0), (-1, 0), colors.HexColor("#1a3a5c")),
        ("TEXTCOLOR",    (0, 0), (-1, 0), colors.white),
        ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, 0), 8),
        ("ALIGN",        (0, 0), (-1, 0), "CENTER"),
        ("VALIGN",       (0, 0), (-1, 0), "MIDDLE"),
        ("TOPPADDING",   (0, 0), (-1, 0), 5),
        ("BOTTOMPADDING",(0, 0), (-1, 0), 5),
        # Data rows
        ("FONTSIZE",     (0, 1), (-1, -1), 8),
        ("VALIGN",       (0, 1), (-1, -1), "TOP"),
        ("TOPPADDING",   (0, 1), (-1, -1), 3),
        ("BOTTOMPADDING",(0, 1), (-1, -1), 3),
        ("ALIGN",        (0, 1), (-1, -1), "LEFT"),
        ("ALIGN",        (3, 1), (3, -1), "RIGHT"),   # Qty
        ("ALIGN",        (5, 1), (-1, -1), "RIGHT"),  # Harga & Total
        # Alternating row shading
        *[("BACKGROUND", (0, r), (-1, r), colors.HexColor("#f0f4f8"))
          for r in range(2, len(items) + 1, 2)],
        # Summary rows
        ("FONTNAME",     (5, -3), (-1, -1), "Helvetica-Bold"),
        ("LINEABOVE",    (5, -3), (-1, -3), 0.8, colors.HexColor("#1a3a5c")),
        ("BACKGROUND",  (5, -1), (-1, -1), colors.HexColor("#1a3a5c")),
        ("TEXTCOLOR",   (5, -1), (-1, -1), colors.white),
        # Grid
        ("GRID",         (0, 0), (-1, len(items)), 0.4, colors.HexColor("#cccccc")),
        ("LINEBELOW",    (0, -1), (-1, -1), 0.8, colors.HexColor("#1a3a5c")),
    ]))
    story.append(item_tbl)
    story.append(Spacer(1, 6 * mm))

    # ── Terms & Notes ───────────────────────────────────────────────────────
    story.append(Paragraph("Syarat & Catatan", styles["SectionH"]))
    terms = [
        ("Syarat Pembayaran", "Net 30 hari setelah barang diterima & invoice diterbitkan"),
        ("Syarat Pengiriman",  "DDP Gudang Tangerang (ongkos kirim ditanggung penjual)"),
        ("Kemasan",            "Produk harus dikemas sesuai standar pengiriman internasional"),
        ("Dokumen",            "Sertakan Certificate of Analysis (CoA) untuk setiap item"),
        ("Catatan Khusus",     "Harap konfirmasi penerimaan PO ini dalam 2 hari kerja"),
    ]
    term_data = [[Paragraph(f"<b>{k}</b>", styles["BodyText2"]),
                  Paragraph(v, styles["BodyText2"])] for k, v in terms]
    term_tbl = Table(term_data, colWidths=[50 * mm, 125 * mm])
    term_tbl.setStyle(TableStyle([
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING",   (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
        ("LINEBELOW",    (0, 0), (-1, -2), 0.3, colors.HexColor("#eeeeee")),
    ]))
    story.append(term_tbl)
    story.append(Spacer(1, 8 * mm))

    # ── Signatures ──────────────────────────────────────────────────────────
    story.append(hr())
    sig_data = [[
        Paragraph("Disetujui oleh,<br/><br/><br/><br/>"
                  "______________________________<br/>"
                  "<b>Budi Santoso</b><br/>"
                  "Purchasing Manager<br/>"
                  "PT Sentosa Aromatics", styles["BodyText2"]),
        Paragraph("Diterima oleh,<br/><br/><br/><br/>"
                  "______________________________<br/>"
                  "<b>________________________</b><br/>"
                  "Sales Representative<br/>"
                  "PT Multisari Indoprima", styles["BodyText2"]),
    ]]
    sig_tbl = Table(sig_data, colWidths=[87 * mm, 88 * mm])
    sig_tbl.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN",  (0, 0), (-1, -1), "CENTER"),
    ]))
    story.append(sig_tbl)

    doc.build(story)
    print(f"  ✔ Generated: {path}")


# ---------------------------------------------------------------------------
# 2. msds_sample.pdf — Material Safety Data Sheet (LUZI AG)
# ---------------------------------------------------------------------------

def generate_msds_sample(path: str):
    doc = SimpleDocTemplate(
        path, pagesize=A4,
        rightMargin=20 * mm, leftMargin=20 * mm,
        topMargin=18 * mm, bottomMargin=18 * mm,
    )
    styles = base_styles()
    story = []

    def section(title):
        story.append(Spacer(1, 3 * mm))
        story.append(Paragraph(title, styles["SectionH"]))
        story.append(hr(thickness=1.0, color=colors.HexColor("#1a3a5c")))

    def field(label, value):
        story.append(Paragraph(
            f"<b>{label}:</b> {value}", styles["BodyText2"]
        ))

    def bullet(text):
        story.append(Paragraph(f"• {text}", styles["BodyText2"]))

    # ── Document Header ─────────────────────────────────────────────────────
    story.append(Paragraph(
        "MATERIAL SAFETY DATA SHEET (MSDS)", styles["DocTitle"]
    ))
    story.append(Paragraph(
        "Sesuai Regulasi (EC) No. 1907/2006 (REACH) &amp; GHS/SDS Format",
        styles["SubTitle"]
    ))
    story.append(Spacer(1, 2 * mm))
    story.append(hr(thickness=1.5, color=colors.HexColor("#1a3a5c")))
    story.append(Spacer(1, 2 * mm))

    meta_data = [
        ["Nama Produk:",   "Linalool Fragrance Compound LZ-4412",
         "Revisi:",        "3.1"],
        ["Supplier:",      "LUZI AG, Gewerbestrasse 10, 4123 Allschwil, Switzerland",
         "Tanggal SDS:",   "01 Agustus 2026"],
        ["No. CAS:",       "78-70-6 (Linalool ≥95%)",
         "No. EINECS:",    "201-134-4"],
        ["No. SDS:",       "LUZI-MSDS-2026-LZ4412",
         "Versi Bahasa:",  "Indonesia / English"],
    ]
    meta_tbl = Table(meta_data, colWidths=[32*mm, 75*mm, 22*mm, 46*mm])
    meta_tbl.setStyle(TableStyle([
        ("FONTSIZE",     (0, 0), (-1, -1), 8),
        ("FONTNAME",     (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",     (2, 0), (2, -1), "Helvetica-Bold"),
        ("BACKGROUND",   (0, 0), (-1, -1), colors.HexColor("#f0f4f8")),
        ("GRID",         (0, 0), (-1, -1), 0.3, colors.HexColor("#cccccc")),
        ("TOPPADDING",   (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
        ("LEFTPADDING",  (0, 0), (-1, -1), 5),
    ]))
    story.append(meta_tbl)

    # ── Section 1 ───────────────────────────────────────────────────────────
    section("BAGIAN 1 — Identifikasi Zat/Campuran dan Perusahaan")
    field("Nama Produk",   "Linalool Fragrance Compound LZ-4412")
    field("Penggunaan",    "Bahan baku parfum, kosmetik, produk perawatan pribadi, dan aromaterapi")
    field("Supplier",      "LUZI AG | Tel: +41 61 486 56 56 | Email: safety@luzi.com")
    field("Darurat",       "Emergency: +41 61 486 56 00 (24 jam) | CHEMTREC: +1-703-527-3887")

    # ── Section 2 ───────────────────────────────────────────────────────────
    section("BAGIAN 2 — Identifikasi Bahaya")
    field("Klasifikasi GHS", "Skin Sensitizer Cat. 1B (H317) | Aquatic Chronic Cat. 3 (H412)")
    story.append(Paragraph("<b>Piktogram Bahaya:</b>", styles["BodyText2"]))
    bullet("GHS07 — Exclamation mark (Iritan/Sensitizer)")
    bullet("GHS09 — Environmental hazard")
    story.append(Paragraph("<b>Pernyataan Bahaya (H-Statements):</b>", styles["BodyText2"]))
    bullet("H317: Dapat menyebabkan reaksi alergi pada kulit")
    bullet("H412: Berbahaya bagi kehidupan perairan dengan efek jangka panjang")
    story.append(Paragraph("<b>Pernyataan Tindakan Pencegahan (P-Statements):</b>", styles["BodyText2"]))
    bullet("P261: Hindari menghirup uap/semprotan")
    bullet("P272: Pakaian kerja yang terkontaminasi tidak boleh dibawa ke luar tempat kerja")
    bullet("P273: Hindari pelepasan ke lingkungan")
    bullet("P280: Gunakan sarung tangan pelindung/pakaian pelindung/alat pelindung mata")
    bullet("P302+P352: JIKA TERKENA KULIT: Cuci dengan air dan sabun dalam jumlah banyak")
    bullet("P333+P313: Jika terjadi iritasi kulit atau ruam: Cari pertolongan medis")
    bullet("P501: Buang isi/wadah sesuai peraturan setempat")

    # ── Section 3 ───────────────────────────────────────────────────────────
    section("BAGIAN 3 — Komposisi / Informasi Bahan")
    comp_data = [
        ["Nama Bahan",     "No. CAS",  "Konsentrasi", "Klasifikasi GHS"],
        ["Linalool",       "78-70-6",  "≥ 95%",       "Skin Sens. 1B; H317\nAquatic Chronic 3; H412"],
        ["Linalyl Acetate","115-95-7", "2 – 4%",      "Skin Sens. 1B; H317"],
        ["α-Terpineol",    "98-55-5",  "≤ 1%",        "Eye Irrit. 2; H319"],
        ["Geraniol",       "106-24-1", "≤ 0.5%",      "Skin Sens. 1A; H317"],
    ]
    comp_tbl = Table(comp_data, colWidths=[48*mm, 25*mm, 25*mm, 77*mm])
    comp_tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), colors.HexColor("#1a3a5c")),
        ("TEXTCOLOR",    (0, 0), (-1, 0), colors.white),
        ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 8),
        ("GRID",         (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING",   (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
        ("LEFTPADDING",  (0, 0), (-1, -1), 4),
        *[("BACKGROUND", (0, r), (-1, r), colors.HexColor("#f7f9fc"))
          for r in range(2, 5, 2)],
    ]))
    story.append(comp_tbl)

    # ── Section 4 ───────────────────────────────────────────────────────────
    section("BAGIAN 4 — Tindakan Pertolongan Pertama")
    story.append(Paragraph("<b>Kontak Mata:</b>", styles["BodyText2"]))
    bullet("Segera bilas dengan air bersih mengalir selama minimal 15 menit sambil kelopak mata dibuka lebar.")
    bullet("Lepaskan lensa kontak jika ada dan mudah dilakukan.")
    bullet("Segera cari pertolongan medis jika iritasi berlanjut.")
    story.append(Paragraph("<b>Kontak Kulit:</b>", styles["BodyText2"]))
    bullet("Segera cuci kulit yang terkena dengan air dan sabun yang banyak selama minimal 10 menit.")
    bullet("Lepaskan pakaian dan sepatu yang terkontaminasi.")
    bullet("Jika terjadi iritasi atau ruam kulit, hubungi dokter.")
    story.append(Paragraph("<b>Terhirup:</b>", styles["BodyText2"]))
    bullet("Pindahkan korban ke udara segar. Istirahat dalam posisi nyaman untuk bernapas.")
    bullet("Jika mengalami kesulitan bernapas, berikan oksigen. Cari pertolongan medis segera.")
    story.append(Paragraph("<b>Tertelan:</b>", styles["BodyText2"]))
    bullet("JANGAN dimuntahkan. Bilas mulut dengan air.")
    bullet("Jika korban sadar, berikan 1-2 gelas air minum.")
    bullet("Segera hubungi dokter atau Poison Control Center.")
    field("Catatan untuk Dokter", "Tidak ada antidot spesifik. Lakukan perawatan simptomatik.")

    # ── Section 5 ───────────────────────────────────────────────────────────
    section("BAGIAN 5 — Tindakan Pemadam Kebakaran")
    field("Titik Nyala (Flash Point)", "77°C (171°F) — Metode: Pensky-Martens Closed Cup")
    field("Suhu Penyulutan",           "235°C")
    field("Batas Mudah Terbakar",      "LEL: 0.9% | UEL: 6.5% (estimasi)")
    field("Media Pemadam Api",         "CO₂, serbuk kimia kering, busa (foam), semprotan air halus (kabut)")
    field("Media yang Dilarang",       "Jangan gunakan semprotan air bertekanan penuh langsung")
    bullet("Gunakan SCBA (Self-Contained Breathing Apparatus) dan pakaian pelindung lengkap saat memadamkan.")
    bullet("Dinginkan wadah yang terpapar api dengan semprotan air.")
    bullet("Gas/uap yang timbul akibat kebakaran dapat menyebabkan iritasi saluran pernapasan.")

    # ── Section 6 ───────────────────────────────────────────────────────────
    section("BAGIAN 6 — Tindakan Penanggulangan Tumpahan")
    story.append(Paragraph("<b>Tindakan Perlindungan Pribadi:</b>", styles["BodyText2"]))
    bullet("Gunakan APD lengkap: sarung tangan nitrile, kacamata pelindung, masker N95.")
    bullet("Jauhkan dari sumber panas dan nyala api — produk mudah terbakar.")
    bullet("Pastikan ventilasi memadai di area tumpahan.")
    story.append(Paragraph("<b>Metode Pembersihan:</b>", styles["BodyText2"]))
    bullet("Serap tumpahan dengan material inert (pasir, vermikulit, atau tanah liat).")
    bullet("Kumpulkan material yang terserap ke dalam wadah berlabel untuk pembuangan.")
    bullet("Jangan membuang produk ke saluran air, tanah, atau lingkungan perairan.")
    bullet("Bersihkan sisa kontaminasi dengan air dan detergen.")

    # ── Section 7 ───────────────────────────────────────────────────────────
    section("BAGIAN 7 — Penanganan dan Penyimpanan")
    story.append(Paragraph("<b>Penanganan:</b>", styles["BodyText2"]))
    bullet("Hindari kontak dengan kulit, mata, dan pakaian.")
    bullet("Hindari menghirup uap atau semprotan.")
    bullet("Gunakan hanya di area berventilasi baik.")
    bullet("Jauhkan dari sumber panas, api, dan bahan pengoksidasi.")
    story.append(Paragraph("<b>Penyimpanan:</b>", styles["BodyText2"]))
    field("Suhu Penyimpanan",   "15°C – 25°C (simpan di tempat sejuk dan kering)")
    field("Wadah",              "Wadah asli tertutup rapat; hindari wadah logam reaktif")
    field("Kondisi Khusus",     "Jauhkan dari cahaya matahari langsung dan sumber panas")
    field("Masa Simpan",        "24 bulan dari tanggal produksi jika disimpan sesuai anjuran")

    # ── Section 8 ───────────────────────────────────────────────────────────
    section("BAGIAN 8 — Pengendalian Paparan / Perlindungan Diri")
    field("NAB (Nilai Ambang Batas)", "Linalool: belum ditetapkan secara nasional — gunakan 10 ppm (TWA) sebagai panduan")
    field("Metode Pengukuran",        "NIOSH 1501 (Hydrocarbons, Aromatic)")
    story.append(Paragraph("<b>Alat Pelindung Diri (APD):</b>", styles["BodyText2"]))
    bullet("Pernapasan: Masker half-face dengan filter kombinasi A2/P2 jika ventilasi tidak memadai")
    bullet("Tangan: Sarung tangan nitrile ≥0.2mm (ganti setiap 2 jam atau saat rusak)")
    bullet("Mata/Wajah: Kacamata pelindung anti-percik / face shield")
    bullet("Tubuh: Pakaian kerja tahan kimia; hindari pakaian yang dapat menyerap cairan")

    # ── Section 9 ───────────────────────────────────────────────────────────
    section("BAGIAN 9 — Sifat Fisika dan Kimia")
    props = [
        ("Penampilan Fisik",         "Cairan jernih, tidak berwarna hingga sedikit kuning pucat"),
        ("Bau",                      "Floral segar, mirip bunga lavender dengan nuansa kayu"),
        ("pH (larutan 1%)",          "Tidak berlaku (produk tidak larut sempurna dalam air)"),
        ("Titik Didih",              "198 – 199°C pada 1 atm"),
        ("Titik Lebur",              "< -20°C"),
        ("Titik Nyala (Flash Point)","77°C (Closed Cup, Pensky-Martens)"),
        ("Kelarutan dalam Air",      "1.7 g/L pada 20°C (sedikit larut)"),
        ("Kelarutan dalam Pelarut",  "Larut sempurna dalam etanol, dietil eter, kloroform"),
        ("Densitas",                 "0.858 – 0.868 g/mL pada 20°C"),
        ("Indeks Bias",              "1.462 – 1.466 pada 20°C"),
        ("Tekanan Uap",              "0.16 hPa pada 20°C"),
        ("Koefisien Partisi (Log P)", "2.97 (oktanol/air)"),
    ]
    prop_data = [[Paragraph(f"<b>{k}</b>", styles["BodyText2"]),
                  Paragraph(v, styles["BodyText2"])] for k, v in props]
    prop_tbl = Table(prop_data, colWidths=[55*mm, 120*mm])
    prop_tbl.setStyle(TableStyle([
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING",   (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 2),
        ("LINEBELOW",    (0, 0), (-1, -2), 0.3, colors.HexColor("#eeeeee")),
        *[("BACKGROUND", (0, r), (-1, r), colors.HexColor("#f7f9fc"))
          for r in range(0, len(props), 2)],
    ]))
    story.append(prop_tbl)

    # ── Section 11 ──────────────────────────────────────────────────────────
    section("BAGIAN 11 — Informasi Toksikologi")
    field("Toksisitas Akut Oral (LD50)",   "> 2.790 mg/kg (tikus) — Kategori: Tidak diklasifikasikan")
    field("Toksisitas Akut Dermal (LD50)", "> 5.000 mg/kg (kelinci) — Kategori: Tidak diklasifikasikan")
    field("Toksisitas Inhalasi",           "LC50 (uap, 4 jam, tikus): > 5 mg/L — Tidak diklasifikasikan")
    field("Iritasi Kulit",                 "Dapat menyebabkan sensitisasi kulit pada paparan berulang (H317)")
    field("Iritasi Mata",                  "Sedikit iritan; tidak menyebabkan kerusakan mata permanen")
    field("Sensitivitas",                  "Diketahui sebagai skin sensitizer; uji LLNA positif pada 5%")
    field("Karsinogenisitas",              "Tidak diklasifikasikan sebagai karsinogen (IARC/NTP/OSHA)")
    field("Efek Reproduksi",              "Tidak ada bukti efek teratogenik atau reproduktif yang signifikan")

    # ── Section 12 ──────────────────────────────────────────────────────────
    section("BAGIAN 12 — Informasi Ekologi")
    field("Ekotoksisitas Akut (Ikan)",       "LC50 (Oncorhynchus mykiss, 96 jam): 27 mg/L")
    field("Ekotoksisitas Akut (Daphnia)",    "EC50 (Daphnia magna, 48 jam): 16 mg/L")
    field("Ekotoksisitas (Alga)",            "ErC50 (Pseudokirchneriella subcapitata, 72 jam): 8.6 mg/L")
    field("Persistensi / Degradabilitas",   "Mudah terbiodegradasi: >70% dalam 28 hari (OECD 301B)")
    field("Potensi Bioakumulasi",            "Log Pow = 2.97 — Potensi bioakumulasi rendah")
    field("Mobilitas dalam Tanah",          "Sedikit mobile; dapat teradsorpsi pada tanah organik")

    # ── Section 13 ──────────────────────────────────────────────────────────
    section("BAGIAN 13 — Pertimbangan Pembuangan")
    bullet("Buang sesuai peraturan nasional/daerah yang berlaku untuk limbah bahan kimia.")
    bullet("Jangan membuang ke saluran pembuangan umum, sungai, atau tanah.")
    bullet("Kode limbah EU: 14 06 03* (pelarut organik lainnya, larutan pencuci, dan cairan induk)")
    bullet("Gunakan jasa pengolahan limbah B3 berlisensi untuk pembuangan.")
    bullet("Wadah kosong: bersihkan tiga kali sebelum daur ulang atau pembuangan.")

    # ── Section 15 ──────────────────────────────────────────────────────────
    section("BAGIAN 15 — Informasi Regulasi")
    field("Regulasi EU",         "REACH (EC 1907/2006), CLP (EC 1272/2008), Cosmetics Regulation (EC 1223/2009)")
    field("IFRA",                "Linalool: Tercantum dalam IFRA Transparency List — termasuk dalam 26 allergen EU")
    field("Indonesia",           "Peraturan BPOM No. 23 Tahun 2019 tentang Persyaratan Teknis Bahan Kosmetika")
    field("Status TSCA",         "Terdaftar dalam TSCA Inventory (USA)")
    field("Persyaratan Labeling", "Label wajib mencantumkan simbol GHS07, GHS09, H317, H412, dan P-statements terkait")

    # ── Section 16 ──────────────────────────────────────────────────────────
    section("BAGIAN 16 — Informasi Lainnya")
    field("Tanggal Penerbitan",  "15 Januari 2024")
    field("Tanggal Revisi",      "01 Agustus 2026 (Revisi 3.1)")
    field("Alasan Revisi",       "Pembaruan data toksikologi Bagian 11 dan penambahan regulasi Indonesia")
    field("Disiapkan oleh",      "Regulatory Affairs Dept. — LUZI AG, Switzerland")
    field("Disclaimer",          "Informasi dalam dokumen ini diyakini akurat pada saat penerbitan. "
                                 "Pengguna bertanggung jawab untuk memastikan kesesuaian penggunaan "
                                 "produk dengan peraturan yang berlaku di wilayahnya.")
    story.append(Spacer(1, 4 * mm))
    story.append(hr(thickness=1.5, color=colors.HexColor("#1a3a5c")))
    story.append(Paragraph(
        "Dokumen ini merupakan Material Safety Data Sheet resmi yang diterbitkan oleh LUZI AG. "
        "Hak cipta dilindungi. Dilarang memperbanyak tanpa izin tertulis.",
        ParagraphStyle("Footer", parent=styles["SmallLabel"], alignment=TA_CENTER)
    ))

    doc.build(story)
    print(f"  ✔ Generated: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating sample PDF files...")
    generate_sample_po(os.path.join(OUTPUT_DIR, "sample_po.pdf"))
    generate_msds_sample(os.path.join(OUTPUT_DIR, "sample_docs.pdf"))
    print("\nDone! Files saved to:", OUTPUT_DIR)
