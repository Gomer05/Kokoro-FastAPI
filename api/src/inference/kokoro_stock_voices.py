"""Voice packs embedded in the pinned Kokoro GGUF conversion."""

import hashlib

import aiofiles

# Git LFS SHA-256 values from hexgrad/Kokoro-82M at f3ff3571791e39611d31c381e3a41a3af07b4987.
STOCK_VOICE_SHA256 = {
    "af_alloy": "6d877149dd8b348fbad12e5845b7e43d975390e9f3b68a811d1d86168bef5aa3",
    "af_aoede": "c03bd1a4c3716c2d8eaa3d50022f62d5c31cfbd6e15933a00b17fefe13841cc4",
    "af_bella": "8cb64e02fcc8de0327a8e13817e49c76c945ecf0052ceac97d3081480e8e48d6",
    "af_heart": "0ab5709b8ffab19bfd849cd11d98f75b60af7733253ad0d67b12382a102cb4ff",
    "af_jessica": "cdfdccb8cc975aa34ee6b89642963b0064237675de0e41a30ae64cc958dd4e87",
    "af_kore": "8bfbc512321c3db49dff984ac675fa5ac7eaed5a96cc31104d3a9080e179d69d",
    "af_nicole": "c5561808bcf5250fe8c5f5de32caf2d94f27e57e95befdb098c5c85991d4c5da",
    "af_nova": "e0233676ddc21908c37a1f102f6b88a59e4e5c1bd764983616eb9eda629dbcd2",
    "af_river": "e149459bd9c084416b74756b9bd3418256a8b839088abb07d463730c369dab8f",
    "af_sarah": "49bd364ea3be9eb3e9685e8f9a15448c4883112a7c0ff7ab139fa4088b08cef9",
    "af_sky": "c799548aed06e0cb0d655a85a01b48e7f10484d71663f9a3045a5b9362e8512c",
    "am_adam": "ced7e284aba12472891be1da3ab34db84cc05cc02b5889535796dbf2d8b0cb34",
    "am_echo": "8bcfdc852bc985fb45c396c561e571ffb9183930071f962f1b50df5c97b161e8",
    "am_eric": "ada66f0eefff34ec921b1d7474d7ac8bec00cd863c170f1c534916e9b8212aae",
    "am_fenrir": "98e507eca1db08230ae3b6232d59c10aec9630022d19accac4f5d12fcec3c37a",
    "am_liam": "c82550757ddb31308b97f30040dda8c2d609a9e2de6135848d0a948368138518",
    "am_michael": "9a443b79a4b22489a5b0ab7c651a0bcd1a30bef675c28333f06971abbd47bd37",
    "am_onyx": "e8452be16cd0f6da7b4579eaf7b1e4506e92524882053d86d72b96b9a7fed584",
    "am_puck": "dd1d8973f4ce4b7d8ae407c77a435f485dabc052081b80ea75c4f30b84f36223",
    "am_santa": "7f2f7582fa2b1f160e90aafe6d0b442a685e773608b6667e545d743b073e97a7",
    "bf_alice": "d292651b6af6c0d81705c2580dcb4463fccc0ff7b8d618a471dbb4e45655b3f3",
    "bf_emma": "d0a423deabf4a52b4f49318c51742c54e21bb89bbbe9a12141e7758ddb5da701",
    "bf_isabella": "cdd4c37003805104d1d08fb1e05855c8fb2c68de24ca6e71f264a30aaa59eefd",
    "bf_lily": "6e09c2e481e2d53004d7e5ae7d3a325369e130a6f45c35a6002de75084be9285",
    "bm_daniel": "fc3fce4e9c12ed4dbc8fa9680cfe51ee190a96444ce7c3ad647549a30823fc5d",
    "bm_fable": "d44935f3135257a9064df99f007fc1342ff1aa767552b4a4fa4c3b2e6e59079c",
    "bm_george": "f1bc812213dc59774769e5c80004b13eeb79bd78130b11b2d7f934542dab811b",
    "bm_lewis": "b5204750dcba01029d2ac9cec17aec3b20a6d64073c579d694a23cb40effbd0e",
    "ef_dora": "d9d69b0f8a2b87a345f269d89639f89dfbd1a6c9da0c498ae36dd34afcf35530",
    "em_alex": "5eac53f767c3f31a081918ba531969aea850bed18fe56419b804d642c6973431",
    "em_santa": "aa8620cb96cec705823efca0d956a63e158e09ad41aca934d354b7f0778f63cb",
    "ff_siwis": "8073bf2d2c4b9543a90f2f0fd2144de4ed157e2d4b79ddeb0d5123066171fbc9",
    "hf_alpha": "06906fe05746d13a79c5c01e21fd7233b05027221a933c9ada650f5aafc8f044",
    "hf_beta": "63c0a1a6272e98d43f4511bba40e30dd9c8ceaf5f39af869509b9f51a71c503e",
    "hm_omega": "b55f02a8e8483fffe0afa566e7d22ed8013acf47ad4f6bbee2795a840155703e",
    "hm_psi": "2f0f055cea4f1083f4ef127ece48d71606347f6557dbb961c0ca5740a2da485b",
    "if_sara": "6c0b253b955fe32f1a1a86006aebe83d050ea95afd0e7be15182f087deedbf55",
    "im_nicola": "234ed06648649f9bd874b37508ea17560b9c993ef85b4ddb3e3a71e062bd2c12",
    "jf_alpha": "1bf4c9dc69e45ee46183b071f4db766349aac5592acbcfeaf051018048a5d787",
    "jf_gongitsune": "1b171917f18f351e65f2bf9657700cd6bfec4e65589c297525b9cf3c20105770",
    "jf_nezumi": "d83f007a7f01783b77014561a7d493d327a0210e143440e91c9b697590d27661",
    "jf_tebukuro": "0d6917904438aec85f73a6fa1f7ac2be6481aae47c697834936930a91796c576",
    "jm_kumo": "98340afd68b1cee84fe0cd95528cfa6d4b39e416aa75a9df64049d52c8b55896",
    "pf_dora": "07e4ff987c5d5a8c3995efd15cc4f0db7c4c15e881b198d8ab7f67ecf51f5eb7",
    "pm_alex": "cf0ba8c573c2480fc54123683a35cf1e2ae130428e441eb91f9149bdb188a526",
    "pm_santa": "d42103169c5c872abbafb9129133af7e942bb9d272c3cc3b95c203e7d7198c29",
    "zf_xiaobei": "9b76be63dab4f4f96962030acc0126a9aee9728608fbbe115e2b58a2bd504df6",
    "zf_xiaoni": "95b49f169bf1640f4f43c25e13daa39f7b98d15d00823e83ef5c14b3ced59e49",
    "zf_xiaoxiao": "cfaf6f2ded1ee56f1ff94fcd2b0e6cdf32e5b794bdc05b44e7439d44aef5887c",
    "zf_xiaoyi": "b5235dbaeef85a4c613bf78af9a88ff63c25bac5f26ba77e36186d8b7ebf05e2",
    "zm_yunjian": "76cbf8bad35901d011d9628a2fdceb7b4f1f127e7a3269cb393b3941eb7fc417",
    "zm_yunxi": "dbe6e1ce7c3dbaf2f5667432947b638b1c6831ccbe154c4610dbcc44f431e27b",
    "zm_yunxia": "bb2b03b08e84d64e1214440ce3b624987fac177f2eeb5bab8571799a3d980acd",
    "zm_yunyang": "5238ac22e0c7f8b6cdd2eddd6e444b8a700b73c4674d9a047d59a94ff96379a2",
}


async def is_stock_voice(name: str, path: str) -> bool:
    expected = STOCK_VOICE_SHA256.get(name)
    if expected is None:
        return False
    try:
        async with aiofiles.open(path, "rb") as voice:
            # All bundled packs are about 512 KiB. Do not read an unbounded replacement.
            data = await voice.read(1024 * 1024)
    except OSError:
        return False
    return hashlib.sha256(data).hexdigest() == expected
