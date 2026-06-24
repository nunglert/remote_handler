pip install uv
uv pip compile pyproject.toml \
    --extra dev --extra mace_freeze \
    --extra clinamen --extra quests --extra neuralil \
    -o requirements-lock.txt
