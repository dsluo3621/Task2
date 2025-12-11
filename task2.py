import csv
import json
import logging
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set
import matplotlib.pyplot as plt
import networkx as nx

# ====================== 1. Configuration and Logging Setup ======================
# Configure logging to record user operations and algorithm runtime status
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('co_purchase_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ====================== 2. Core Data Structure: Weighted Undirected Graph ======================
class WeightedUndirectedGraph:
    """
    Weighted undirected graph for storing product co-purchase relationships
    - Nodes: Product names
    - Edges: Co-purchase relationships between products
    - Edge weights: Co-purchase frequency (count)
    """

    def __init__(self):
        # Adjacency list: {productA: {productB: co_purchase_count, productC: co_purchase_count}, ...}
        self.adjacency_list: Dict[str, Dict[str, int]] = defaultdict(dict)
        # Total purchase frequency statistics for each product
        self.product_frequency: Counter = Counter()
        # Product categories (for filtering functionality)
        self.product_categories: Dict[str, str] = self._init_product_categories()

    def _init_product_categories(self) -> Dict[str, str]:
        """Initialize product categories (manually labeled core products to support filtering)"""
        categories = {
            # Dairy products
            'whole milk': 'dairy',
            'yogurt': 'dairy',
            'whipped/sour cream': 'dairy',
            'butter': 'dairy',
            'cheese': 'dairy',
            # Vegetables
            'other vegetables': 'vegetables',
            'root vegetables': 'vegetables',
            'carrots': 'vegetables',
            'tomatoes': 'vegetables',
            # Fruits
            'tropical fruit': 'fruits',
            'pip fruit': 'fruits',
            'citrus fruit': 'fruits',
            'grapes': 'fruits',
            'berries': 'fruits',
            # Bakery
            'rolls/buns': 'bakery',
            'brown bread': 'bakery',
            'white bread': 'bakery',
            'pastry': 'bakery',
            # Drinks
            'soda': 'drinks',
            'bottled water': 'drinks',
            'canned beer': 'drinks',
            'bottled beer': 'drinks',
            'coffee': 'drinks',
            'tea': 'drinks',
            # Meat
            'sausage': 'meat',
            'frankfurter': 'meat',
            'pork': 'meat',
            'beef': 'meat',
            'chicken': 'meat'
        }
        return categories

    def add_transaction(self, items: List[str]) -> None:
        """
        Add a new transaction and update the graph structure
        :param items: List of products in a single transaction (after deduplication)
        """
        if len(items) < 2:
            # Single-product transaction: only update product purchase frequency, no co-purchase relationships
            for item in items:
                self.product_frequency[item] += 1
            logger.info(f"Single-product transaction, only updating product frequency: {items}")
            return

        # 1. Update total product purchase frequency
        for item in items:
            self.product_frequency[item] += 1

        # 2. Generate all unordered product pairs and update co-purchase counts
        sorted_items = sorted(items)  # Sort to ensure unordered product pairs (A,B and B,A are treated as the same)
        for i in range(len(sorted_items)):
            item1 = sorted_items[i]
            for j in range(i + 1, len(sorted_items)):
                item2 = sorted_items[j]
                # Update adjacency edge for item1
                if item2 in self.adjacency_list[item1]:
                    self.adjacency_list[item1][item2] += 1
                else:
                    self.adjacency_list[item1][item2] = 1
                # Update adjacency edge for item2 (undirected graph symmetry)
                if item1 in self.adjacency_list[item2]:
                    self.adjacency_list[item2][item1] += 1
                else:
                    self.adjacency_list[item2][item1] = 1

        logger.info(f"Processed transaction, updated co-purchase relationships: {items}")

    def get_top_co_purchase(self, target_product: str, top_n: int = 5) -> List[Tuple[str, int]]:
        """
        Query the TopN products most frequently co-purchased with the target product
        :param target_product: Name of the target product
        :param top_n: Number of results to return
        :return: [(product_name, co_purchase_count), ...] (sorted descending by count)
        """
        if target_product not in self.adjacency_list:
            logger.warning(f"No co-purchase records found for product: {target_product}")
            return []

        # Sort by co-purchase count in descending order
        co_purchase_items = sorted(
            self.adjacency_list[target_product].items(),
            key=lambda x: x[1],
            reverse=True
        )
        return co_purchase_items[:top_n]

    def get_top3_product_pairs(self) -> List[Tuple[Tuple[str, str], int]]:
        """
        Get the Top3 most popular product combinations (product pairs with highest co-purchase counts)
        :return: [( (productA, productB), co_purchase_count ), ...]
        """
        all_pairs = []
        processed_pairs = set()  # Avoid duplicate counting (A,B and B,A are treated as the same pair)

        for item1, neighbors in self.adjacency_list.items():
            for item2, count in neighbors.items():
                # Ensure product pairs are sorted alphabetically to avoid duplicates
                pair = tuple(sorted([item1, item2]))
                if pair not in processed_pairs:
                    processed_pairs.add(pair)
                    all_pairs.append((pair, count))

        # Sort by co-purchase count in descending order and take top 3
        all_pairs_sorted = sorted(all_pairs, key=lambda x: x[1], reverse=True)
        return all_pairs_sorted[:3]

    def check_co_purchase_relation(self, item1: str, item2: str) -> int:
        """
        Check if two products have a co-purchase relationship, return co-purchase count (0 if none)
        :param item1: First product name
        :param item2: Second product name
        :return: Co-purchase count
        """
        if item1 in self.adjacency_list and item2 in self.adjacency_list[item1]:
            return self.adjacency_list[item1][item2]
        return 0

    def filter_by_category(self, category: str) -> Dict[str, Dict[str, int]]:
        """
        Filter the graph structure by product category, return only products and their co-purchase relationships in the specified category
        :param category: Category name (e.g., dairy/vegetables)
        :return: Filtered adjacency list
        """
        # First get all products in the specified category
        category_products = [p for p, cat in self.product_categories.items() if cat == category]
        if not category_products:
            logger.warning(f"No products found in category: {category}")
            return {}

        # Filter adjacency list: only keep edges between products in the category
        filtered_adj = {}
        for product in category_products:
            if product in self.adjacency_list:
                filtered_neighbors = {
                    neighbor: count
                    for neighbor, count in self.adjacency_list[product].items()
                    if neighbor in category_products
                }
                if filtered_neighbors:
                    filtered_adj[product] = filtered_neighbors

        return filtered_adj

    def get_recommendation(self, input_products: List[str], top_n: int = 5) -> List[Tuple[str, int]]:
        """
        Recommend products most likely to be co-purchased based on input products/product combinations
        :param input_products: List of input products
        :param top_n: Number of recommendations to return
        :return: [(recommended_product, total_co_purchase_count), ...]
        """
        recommendation_scores = Counter()

        for product in input_products:
            if product in self.adjacency_list:
                # Aggregate all co-purchased products and their counts for this product
                for neighbor, count in self.adjacency_list[product].items():
                    # Exclude input products themselves
                    if neighbor not in input_products:
                        recommendation_scores[neighbor] += count

        # Sort by total score in descending order and take top N
        return recommendation_scores.most_common(top_n)


# ====================== 3. Data Loading Module ======================
def load_supermarket_data(file_path: str) -> Dict[str, List[str]]:
    """
    Load supermarket transaction data, grouped by transaction_id (member number + date)
    :param file_path: Path to CSV file
    :return: {transaction_id: [product1, product2, ...], ...}
    """
    transactions = defaultdict(list)

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Extract key fields
                member_id = row['Member_number']
                date = row['Date']
                item = row['itemDescription'].strip()

                # Define transaction_id: member number + date (same member on same day = 1 transaction)
                transaction_id = f"{member_id}_{date}"
                # Add product (deduplicate to avoid repeated products in the same transaction)
                if item not in transactions[transaction_id]:
                    transactions[transaction_id].append(item)

        logger.info(f"Successfully loaded data, total transactions: {len(transactions)}")
        return dict(transactions)

    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        raise


# ====================== 4. Visualization Module ======================
def visualize_product_graph(graph: WeightedUndirectedGraph,
                            top_n_products: int = 10,
                            output_path: str = 'product_co_purchase_graph.png') -> None:
    """
    Visualize product co-purchase relationship graph (only show TopN high-frequency products)
    :param graph: Instance of WeightedUndirectedGraph
    :param top_n_products: Number of high-frequency products to display
    :param output_path: Path to save the visualization image
    """
    # 1. Filter TopN high-frequency products
    top_products = [p for p, _ in graph.product_frequency.most_common(top_n_products)]
    # 2. Build subgraph (only include edges between TopN products)
    G = nx.Graph()

    # Add nodes (size proportional to purchase frequency)
    for product in top_products:
        freq = graph.product_frequency[product]
        G.add_node(product, size=freq / 10)  # Scale node size

    # Add edges (thickness proportional to co-purchase count)
    for product in top_products:
        if product in graph.adjacency_list:
            for neighbor, count in graph.adjacency_list[product].items():
                if neighbor in top_products:
                    G.add_edge(product, neighbor, weight=count / 5)  # Scale edge thickness

    # 3. Plot configuration
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, k=2, iterations=50)  # Layout adjustment

    # Draw nodes (size related to purchase frequency)
    node_sizes = [G.nodes[node]['size'] for node in G.nodes]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                           node_color='lightblue', alpha=0.8, edgecolors='black')

    # Draw edges (thickness related to co-purchase count)
    edges = G.edges()
    edge_widths = [G[edge[0]][edge[1]]['weight'] for edge in edges]
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, edge_color='gray')

    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold')

    # Add edge weight labels (only show weights > 50)
    edge_labels = {(u, v): G[u][v]['weight'] * 5 for u, v in edges if G[u][v]['weight'] * 5 > 50}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)

    plt.title(
        f'Top{top_n_products} Product Co-purchase Relationship Graph (Node size = Purchase count, Edge thickness = Co-purchase count)',
        fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Co-purchase relationship graph saved to: {output_path}")


# ====================== 5. Main Program (CLI Interface) ======================
def main():
    """Main program: Provide CLI interface to support all core functionalities"""
    # 1. Initialize graph structure
    graph = WeightedUndirectedGraph()

    # 2. Load data and build graph
    try:
        # Replace with your dataset path
        data_path = "Supermarket_dataset_PAI.csv"
        transactions = load_supermarket_data(data_path)

        # Iterate through all transactions to build the graph
        for trans_id, items in transactions.items():
            graph.add_transaction(items)

        logger.info("Graph structure built successfully")
    except Exception as e:
        print(f"Failed to load data: {str(e)}")
        return

    # 3. CLI interactive menu
    while True:
        print("\n===== Supermarket Transaction Co-purchase Analysis System =====")
        print("1. Query top co-purchased products for a specific product")
        print("2. Query Top3 most popular product combinations")
        print("3. Check co-purchase relationship between two products")
        print("4. Filter product co-purchase relationships by category")
        print("5. Recommend co-purchased products based on product combinations")
        print("6. Generate product co-purchase relationship visualization")
        print("7. Exit system")

        choice = input("\nPlease enter function number (1-7): ").strip()

        if choice == '1':
            # Function 1: Query Top co-purchased products for a specified product
            target = input("Please enter product name (e.g., whole milk): ").strip()
            top_n = input("Please enter number of results to return (default 5): ").strip()
            top_n = int(top_n) if top_n.isdigit() else 5

            result = graph.get_top_co_purchase(target, top_n)
            if result:
                print(f"\nTop{top_n} products most frequently co-purchased with '{target}':")
                for idx, (item, count) in enumerate(result, 1):
                    print(f"  {idx}. {item:<20} Co-purchase count: {count}")
            else:
                print(f"\nNo co-purchase records found for '{target}'")

        elif choice == '2':
            # Function 2: Query Top3 most popular product combinations
            result = graph.get_top3_product_pairs()
            print("\nTop3 most popular product combinations:")
            for idx, ((item1, item2), count) in enumerate(result, 1):
                print(f"  {idx}. {item1:<20} + {item2:<20} Co-purchase count: {count}")

        elif choice == '3':
            # Function 3: Check co-purchase relationship between two products
            item1 = input("Please enter first product name: ").strip()
            item2 = input("Please enter second product name: ").strip()

            count = graph.check_co_purchase_relation(item1, item2)
            if count > 0:
                print(f"\nCo-purchase count between '{item1}' and '{item2}': {count}")
            else:
                print(f"\nNo co-purchase relationship between '{item1}' and '{item2}'")

        elif choice == '4':
            # Function 4: Filter by category
            print("\nAvailable categories: dairy, vegetables, fruits, bakery, drinks, meat")
            category = input("Please enter category name: ").strip().lower()

            filtered_adj = graph.filter_by_category(category)
            if filtered_adj:
                print(f"\nProduct co-purchase relationships in {category} category:")
                for product, neighbors in filtered_adj.items():
                    print(f"  {product}: {neighbors}")
            else:
                print(f"\nNo co-purchase data found in {category} category")

        elif choice == '5':
            # Function 5: Product recommendation
            input_products = input(
                "Please enter product combination (separated by commas, e.g., whole milk,yogurt): ").strip().split(',')
            input_products = [p.strip() for p in input_products if p.strip()]

            if not input_products:
                print("Please enter valid product names")
                continue

            top_n = input("Please enter number of recommendations (default 5): ").strip()
            top_n = int(top_n) if top_n.isdigit() else 5

            recommendations = graph.get_recommendation(input_products, top_n)
            if recommendations:
                print(f"\nRecommended co-purchased products based on '{','.join(input_products)}':")
                for idx, (item, count) in enumerate(recommendations, 1):
                    print(f"  {idx}. {item:<20} Total co-purchase count: {count}")
            else:
                print("\nNo recommended products found")

        elif choice == '6':
            # Function 6: Visualization
            top_n = input("Please enter number of high-frequency products to display (default 10): ").strip()
            top_n = int(top_n) if top_n.isdigit() else 10

            try:
                visualize_product_graph(graph, top_n)
                print("\nVisualization graph generated successfully!")
            except Exception as e:
                print(f"\nFailed to generate visualization: {str(e)}")

        elif choice == '7':
            # Function 7: Exit
            print("Thank you for using the system! Exiting...")
            logger.info("User exited the system")
            break

        else:
            print("Invalid input, please enter a number between 1 and 7")


# ====================== 6. Test Module ======================
# ====================== 6. Test Module ======================
# ====================== 6. Test Module ======================
def run_tests():
    """Run automated tests to verify core functionalities"""
    # 1. Initialize test graph
    test_graph = WeightedUndirectedGraph()

    # 2. Test transaction data
    test_transactions = [
        ["whole milk", "other vegetables", "rolls/buns"],
        ["whole milk", "yogurt"],
        ["other vegetables", "rolls/buns", "soda"],
        ["whole milk", "other vegetables"],
        ["yogurt", "whole milk", "soda"]
    ]

    # 3. Build test graph
    for trans in test_transactions:
        test_graph.add_transaction(trans)

    # 4. Test case 1: Query co-purchased products for a specified product
    result1 = test_graph.get_top_co_purchase("whole milk", 2)
    result1_sorted = sorted(result1, key=lambda x: (-x[1], x[0]))
    expected1 = [("other vegetables", 2), ("yogurt", 2)]
    expected1_sorted = sorted(expected1, key=lambda x: (-x[1], x[0]))
    assert result1_sorted == expected1_sorted, f"Test case 1 failed: {result1}"

    # Test case 2: Query Top3 product combinations
    result2 = test_graph.get_top3_product_pairs()
    expected2 = [
        (("other vegetables", "rolls/buns"), 2),
        (("other vegetables", "whole milk"), 2),
        (("whole milk", "yogurt"), 2)
    ]

    def sort_pair(pair_tuple):
        """Sort product pairs to ensure consistent comparison"""
        pair, count = pair_tuple
        sorted_pair = tuple(sorted(pair))
        return (-count, sorted_pair[0], sorted_pair[1])

    result2_sorted = sorted(result2, key=sort_pair)
    expected2_sorted = sorted(expected2, key=sort_pair)
    assert result2_sorted == expected2_sorted, f"Test case 2 failed: {result2}"

    # Test case 3: Check co-purchase relationship between two products
    result3 = test_graph.check_co_purchase_relation("whole milk", "soda")
    assert result3 == 1, f"Test case 3 failed: {result3}"

    # Test case 4: Product recommendation
    result4 = test_graph.get_recommendation(["whole milk", "yogurt"], 1)
    expected4 = [("other vegetables", 2)]
    assert result4 == expected4, f"Test case 4 failed: {result4}"

    print("All test cases passed!")
    logger.info("Automated testing completed, all cases passed")


# ====================== Entry Point ======================
if __name__ == "__main__":
    # Run tests first
    run_tests()
    # Start main program
    main()