from surgicalai import mesh_parser


def test_mesh_parser(synthetic_mesh):
    data = mesh_parser.load_mesh(synthetic_mesh)
    assert data["coordinates"].shape[1] == 3
    assert "nasion" in data["landmarks"]
