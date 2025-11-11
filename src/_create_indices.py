# Create DB Indices for Qdrant
# ----------------------------
# To be run once after vectore database creation.
# Required to get artist / entry stats.
#
#   poetry shell
#   python src/_create_indices.py

import asyncio
from core.services import get_qdrant_service

qdrant_service = get_qdrant_service()


async def create_indices():
    await qdrant_service.connect()
    print("\n-------")

    # Create keyword indices for ID fields
    print("\nCreating keyword indices:")
    for index in ["artist_id", "entry_id", "view_id"]:
        result = await qdrant_service.client.create_payload_index(
            collection_name=qdrant_service.collection_name,
            field_name=index,
            field_schema="keyword",
        )
        print(index, "-->", result)

    # Create float index for timestamp field (required for order_by)
    print("\nCreating float index for timestamp:")
    result = await qdrant_service.client.create_payload_index(
        collection_name=qdrant_service.collection_name,
        field_name="upload_timestamp",
        field_schema="float",
    )
    print("upload_timestamp -->", result)

    # Verify indices
    print("\nCollection details:")
    collection_info = await qdrant_service.client.get_collection(
        collection_name=qdrant_service.collection_name
    )
    print(collection_info)


if __name__ == "__main__":
    asyncio.run(create_indices())
