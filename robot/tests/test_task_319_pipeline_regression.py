import unittest

import run_all
from task_319_garbage_sort.pipeline import GarbageSortingPipeline


class Task319PipelineRegressionTests(unittest.TestCase):
    def test_task_319_defaults_to_kuavo_config(self):
        self.assertEqual(
            run_all.run_task_319.__defaults__[0],
            "configs/task_319_kuavo_wheel.yaml",
        )

    def test_resolve_categories_keeps_prediction_separate_from_execution(self):
        pipeline = GarbageSortingPipeline.__new__(GarbageSortingPipeline)
        predicted, execution, expected = GarbageSortingPipeline._resolve_categories(
            pipeline,
            "trash_04",
            "bottle",
        )
        self.assertEqual(predicted, "recyclable")
        self.assertEqual(expected, "kitchen_waste")
        self.assertEqual(execution, "kitchen_waste")

    def test_navigation_complete_accepts_recorded_kuavo_stages(self):
        pipeline = GarbageSortingPipeline.__new__(GarbageSortingPipeline)
        pipeline.robot_type = "kuavo_wheel"
        records = [
            {
                "source": "kuavo_grasp_reachability_search",
                "stage": "grasp_feasible_base",
                "success": True,
            },
            {
                "source": "kuavo_release_lowering",
                "stage": "bin_lower",
                "success": True,
            },
        ]
        self.assertTrue(GarbageSortingPipeline._navigation_complete(pipeline, records))

    def test_kuavo_attach_anchor_prefers_live_grasp_center(self):
        pipeline = GarbageSortingPipeline.__new__(GarbageSortingPipeline)

        class Controller:
            @staticmethod
            def get_grasp_center_pos():
                return [1.0, 2.0, 3.0]

        class Env:
            @staticmethod
            def get_body_position(_):
                return [4.0, 5.0, 6.0]

        pipeline.controller = Controller()
        pipeline.env = Env()
        self.assertEqual(
            GarbageSortingPipeline._kuavo_attach_anchor(pipeline, "trash_01").tolist(),
            [1.0, 2.0, 3.0],
        )


if __name__ == "__main__":
    unittest.main()
