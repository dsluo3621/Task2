import unittest
from task2 import WeightedUndirectedGraph  # 确保主文件名为task2.py，否则修改此处


class TestSupermarketCoPurchaseAnalysis(unittest.TestCase):
    """Task2核心功能单元测试"""

    def setUp(self):
        """初始化测试环境"""
        self.graph = WeightedUndirectedGraph()
        # 测试交易数据
        self.test_transactions = [
            ["whole milk", "other vegetables", "rolls/buns"],
            ["whole milk", "yogurt"],
            ["other vegetables", "rolls/buns", "soda"],
            ["whole milk", "other vegetables"],
            ["yogurt", "whole milk", "soda"],
            ["bread"]  # 单商品交易
        ]
        # 构建测试图
        for trans in self.test_transactions:
            self.graph.add_transaction(trans)

    def test_get_top_co_purchase(self):
        """测试：查询指定商品的Top共购商品"""
        # 测试存在共购记录的商品
        result = self.graph.get_top_co_purchase("whole milk", 3)
        self.assertEqual(result, [("other vegetables", 2), ("yogurt", 2), ("rolls/buns", 1)])

        # 测试无共购记录的商品
        result_empty = self.graph.get_top_co_purchase("bread")
        self.assertEqual(result_empty, [])

    def test_get_top3_product_pairs(self):
        """测试：查询Top3热门商品组合"""
        result = self.graph.get_top3_product_pairs()
        # 修正预期值：匹配实际共购次数统计
        expected = [
            (("other vegetables", "rolls/buns"), 2),
            (("other vegetables", "whole milk"), 2),
            (("whole milk", "yogurt"), 2)
        ]
        self.assertEqual(result, expected)

    def test_check_co_purchase_relation(self):
        """测试：检查两商品的共购关系"""
        # 存在共购关系
        self.assertEqual(self.graph.check_co_purchase_relation("whole milk", "soda"), 1)
        # 不存在共购关系
        self.assertEqual(self.graph.check_co_purchase_relation("bread", "milk"), 0)

    def test_filter_by_category(self):
        """测试：按分类过滤商品"""
        # 测试乳制品分类
        dairy_filtered = self.graph.filter_by_category("dairy")
        self.assertIn("whole milk", dairy_filtered)
        self.assertIn("yogurt", dairy_filtered["whole milk"])

        # 测试不存在的分类
        empty_filtered = self.graph.filter_by_category("seafood")
        self.assertEqual(empty_filtered, {})

    def test_get_recommendation(self):
        """测试：商品推荐功能"""
        # 单商品推荐
        single_reco = self.graph.get_recommendation(["whole milk"], 2)
        self.assertEqual(single_reco, [("other vegetables", 2), ("yogurt", 2)])

        # 多商品推荐（修正预期值）
        multi_reco = self.graph.get_recommendation(["whole milk", "yogurt"], 1)
        self.assertEqual(multi_reco, [("other vegetables", 2)])

        # 空输入
        empty_reco = self.graph.get_recommendation([], 1)
        self.assertEqual(empty_reco, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)