from surgicalai import mesh_parser, flap_designer


def test_flap_designer(synthetic_mesh):
    mesh_data = mesh_parser.load_mesh(synthetic_mesh)
    plan = flap_designer.design(mesh_data, mesh_data["coordinates"].mean(axis=0))
    assert "flap_vectors" in plan
    assert 0 <= plan["success_probability"] <= 1
