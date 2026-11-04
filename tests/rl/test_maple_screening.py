"""The fixed Stage 0 selection rule must not drift at boundaries or ties."""

import unittest

from embodied_ai_lab.eval.maple_stage0 import select_variant, variants


class MapleScreeningTest(unittest.TestCase):
    def test_closed_interval_and_yaml_order_tie_break(self):
        candidates = [
            {"name": "below", "success_rate": .19},
            {"name": "first", "success_rate": .375},
            {"name": "second", "success_rate": .625},
            {"name": "above", "success_rate": .71},
        ]
        self.assertEqual(select_variant(candidates)["name"], "first")
        for rate in (.2, .7):
            with self.subTest(rate=rate):
                self.assertEqual(select_variant([{"success_rate": rate}])["success_rate"], rate)
        self.assertIsNone(select_variant([candidates[0], candidates[-1]]))

    def test_official_nine_candidates_and_shared_instruction(self):
        candidates = variants()
        self.assertEqual(len(candidates), 9)
        self.assertEqual(candidates[0]["name"], "droid_pnp_srl_gr00t_billiard_hall")
        self.assertEqual(candidates[-1]["name"], "droid_pnp_srl_gr00t_mug")
        self.assertEqual(candidates[0]["environment"]["hdr"], "billiard_hall_robolab")
        self.assertEqual(
            candidates[0]["environment_builder"]["language_instruction"],
            "Pick up the Rubik's cube and place it in the bowl.",
        )
        self.assertEqual(candidates[1]["environment"]["pick_up_object"], "blue_block_basic_robolab")
        self.assertEqual(
            candidates[1]["environment_builder"]["language_instruction"],
            "Pick up the blue block and place it in the bowl.",
        )


if __name__ == "__main__":
    unittest.main()
