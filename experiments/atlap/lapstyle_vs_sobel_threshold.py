from cast import ExperimentRun, get_device, Vgg16, perform_transfer, save_image

STYLE_CONTENT_PAIRS = [
    ("abstract", "details1"),
    ("candy", "details2"),
    ("la_muse", "details3"),
    ("la_muse", "portrait3"),
    ("modern_abstract", "details1"),
    ("abstract", "bottles"),
    ("abstract", "details3"),
    ("abstract", "details4"),
    ("abstract", "photo1"),
    ("abstract", "portrait1"),
    ("abstract", "portrait3"),
    ("aqua_puantilism_2", "details3"),
    ("aqua_puantilism_2", "portrait3"),
    ("aqua_puantilism_2", "portrait4"),
    ("smallworld", "boy"),
    ("candy", "bottles"),
    ("candy", "details3"),
    ("geometry13", "portrait9"),
    ("hadrd_geometry", "portrait5"),
    ("hadrd_geometry", "portrait9"),
    ("lowdetail_aqua", "portrait8"),
    ("lowdetail_aqua", "portrait9")
]

exp = ExperimentRun('lapstyle_vs_sobel_gs_4')

exp.dump_sources()

device = get_device()

vgg = Vgg16().to(device)

fails = []

for s, c in STYLE_CONTENT_PAIRS:
    item_name = f'{s}__{c}'

    if (exp.results_dir / item_name).exists():
        continue

    item_dir = exp.make_item_dir(item_name)
    content = exp.load_content(c, 512, item_dir)
    style = exp.load_style(s, 512, item_dir)

    without_edge = perform_transfer(
        vgg, content, style,
        1e8,
        device=device, init='cpn'
    )

    save_image(without_edge, item_dir / 'noedge.jpg')

    for ew in [1, 5, 10]:
        res = perform_transfer(
            vgg, content, style,
            1e4,
            'atlap', dict(detector="sobel_gs", normalize=False, reduction="mse"), ew,
            device=device, init='cpn'
        )
        res_path = item_dir / f'qt_e={ew:g}.jpg'
        save_image(res, res_path)


    for lew in [50, 100, 200]:
        lapstyle_1 = perform_transfer(
            vgg, content, style,
            1e4,
            'atlap', dict(detector="lapstyle", normalize=False, reduction="mse"), lew,
            device=device
        )
        save_image(lapstyle_1, item_dir / f'lapstyle_e={lew:06d}.jpg')
