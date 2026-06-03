from actum.core.memory import MemoryStore


def test_memory_store_persists_facts_people_places_and_observations(tmp_path):
    path = tmp_path / "memory.json"
    store = MemoryStore({"path": str(path), "max_episodes": 10})

    store.remember_fact("user_name", "Simon")
    store.remember_person("Simon", "Likes robotics prototypes.")
    store.remember_place("desk", "The laptop dock is on the left side.")
    store.remember_spatial_note("Charging dock is left of the desk.", place="desk")
    obs = store.record_observation("Saw a blue cube on the table.", tags=["table", "cube"])

    assert obs.id == "mem-2"
    assert path.exists()

    reloaded = MemoryStore({"path": str(path)})

    assert reloaded.recall_fact("user_name") == "Simon"
    assert reloaded.people["Simon"]["notes"][-1]["text"] == "Likes robotics prototypes."
    assert reloaded.places["desk"]["notes"][-1]["text"] == "The laptop dock is on the left side."
    assert reloaded.spatial_notes[-1].summary == "Charging dock is left of the desk."
    assert reloaded.recent(1)[0]["id"] == obs.id
    assert "blue cube" in reloaded.context()
