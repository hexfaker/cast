from cast import ExperimentRun, get_device, Vgg16, perform_transfer, save_image

STYLE_CONTENT_PAIRS = [
    ("abstract", "details1"),
    ("candy", "details2"),
    ("la_muse", "details3"),
    ("la_muse", "portrait3"),
    ("modern_abstract", "details1"),
    #
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

exp = ExperimentRun('scaling-1')

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

    for ew in [1e-1, 1e2, 1e3, 1e4]:
        with_scaling = perform_transfer(
            vgg, content, style,
            1e8,
            'atlap', dict(detector="sobel_qt", reduction="mse"), ew,
            device=device, init='cpn'
        )
        save_image(with_scaling, item_dir / f'scaled_{ew:g}.jpg')
