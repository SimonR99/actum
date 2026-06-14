from actum.core.memory import MemoryStore


def test_memory_store_persists_facts_people_places_and_observations(tmp_path):
    path = tmp_path / "memory.json"
    store = MemoryStore({"path": str(path), "max_episodes": 10})

    store.remember_fact("user_name", "Simon")
    store.remember_person("Simon", "Likes robotics prototypes.")
    store.remember_place("desk", "The laptop dock is on the left side.")
    store.remember_spatial_note("Charging dock is left of the desk.", place="desk")
    obs = store.record_observation(
        "Saw a blue cube on the table.", tags=["table", "cube"]
    )

    assert obs.id == "mem-2"
    assert path.exists()

    reloaded = MemoryStore({"path": str(path)})

    assert reloaded.recall_fact("user_name") == "Simon"
    assert reloaded.people["Simon"]["notes"][-1]["text"] == "Likes robotics prototypes."
    assert (
        reloaded.places["desk"]["notes"][-1]["text"]
        == "The laptop dock is on the left side."
    )
    assert reloaded.spatial_notes[-1].summary == "Charging dock is left of the desk."
    assert reloaded.recent(1)[0]["id"] == obs.id
    assert "blue cube" in reloaded.context()


def test_memory_search_ranks_relevant_entries(tmp_path):
    store = MemoryStore({"path": str(tmp_path / "memory.json")})
    store.remember_place("kitchen", "The coffee machine is by the window.")
    store.record_observation("Saw a blue cube on the table.")
    store.record_observation("The charging dock battery is low.")

    coffee = store.search("where is the coffee machine")
    assert coffee and coffee[0][0] == "place"

    cube = store.search("blue cube")
    assert any("blue cube" in text for _, text in cube)

    assert store.search("submarine periscope") == []


def test_query_context_returns_relevant_only(tmp_path):
    store = MemoryStore({"path": str(tmp_path / "memory.json")})
    store.remember_fact("user_name", "Simon")
    store.record_observation("The garage door is open.")

    context = store.context(query="garage door")
    assert "garage door" in context
    assert "user_name" not in context


def test_consolidate_removes_duplicate_records(tmp_path):
    store = MemoryStore({"path": str(tmp_path / "memory.json")})
    store.record_observation("Saw a blue cube on the table.")
    store.record_observation("Saw a blue cube on the table.")
    store.record_observation("Different observation.")

    report = store.consolidate()
    assert report["removed_episodes"] == 1
    assert len(store.episodes) == 2

    reloaded = MemoryStore({"path": str(tmp_path / "memory.json")})
    assert len(reloaded.episodes) == 2
