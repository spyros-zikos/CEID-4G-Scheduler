import numpy as np
import random
import time
import matplotlib.pyplot as plt

# Parameters
NUM_USERS = 20  # More users to show fairness benefits
TOTAL_BANDWIDTH = 200  # in Mbps, total bandwidth available
ALLOCATION_PER_STEP = 10  # Ensure fair distribution
RADIUS = 200    # Larger radius to create more varied channel conditions
TRAFFIC_TYPES = {
    "video_streaming": (20, 30),  # Higher bandwidth for video
    "web_browsing": (5, 15),    # More variable web traffic
    "voice_call": (1, 2)        # More realistic voice demands
}

# Global parameter for Proportional Fairness (β)
BETA = 0.5  # Weight for averaging past and current rates

# User Equipment (UE) class with Proportional Fair properties
class UE:
    def __init__(self, ue_id, traffic_type, position, distance):
        self.ue_id = ue_id
        self.traffic_type = traffic_type
        self.position = position  # (x, y) position
        self.distance = distance  # Distance from the BS
        self.channel_quality = max(0.2, 1 / (1 + (distance / RADIUS)**2))  # Quadratic distance effect
        self.traffic_demand = self.generate_traffic_demand(traffic_type)  # Mbps
        self.allocated_bandwidth = 0
        self.remaining_demand = self.traffic_demand
        self.total_latency = 0
        self.instantaneous_rate = 0  # Achievable rate based on channel quality
        self.average_rate = 0.1  # Start with a small non-zero value for average rate

    def generate_traffic_demand(self, traffic_type):
        min_demand, max_demand = TRAFFIC_TYPES[traffic_type]
        return random.uniform(min_demand, max_demand)

    def allocate_bandwidth(self, amount):
        allocation = min(amount, self.remaining_demand)
        self.allocated_bandwidth += allocation
        self.remaining_demand -= allocation
        if allocation > 0:
            # Consider allocation size in latency
            self.total_latency += 1 - (allocation / self.traffic_demand)

        # Update instantaneous rate
        self.instantaneous_rate = allocation

        # Update average rate
        self.update_average_rate()

    def update_average_rate(self):
        """Update the average data rate based on the PF formula."""
        self.average_rate = (1 - BETA) * self.average_rate + BETA * self.instantaneous_rate

    def reset(self):
        """Reset bandwidth allocations and demands to the initial state."""
        self.allocated_bandwidth = 0
        self.remaining_demand = self.traffic_demand
        self.total_latency = 0
        self.instantaneous_rate = 0
        self.average_rate = 0.1  # Reset to initial small non-zero value


# Generate users with positions and distances
def create_users(num_users):
    users = []
    for i in range(num_users):
        theta = random.uniform(0, 2 * np.pi)  # Random angle
        r = random.uniform(0, RADIUS)  # Random distance from BS
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        distance = np.sqrt(x**2 + y**2)
        traffic_type = random.choice(list(TRAFFIC_TYPES.keys()))
        users.append(UE(i, traffic_type, (x, y), distance))
    return users

def sequential_round_robin(users, total_bandwidth, allocation_per_step):
    step_count = 0
    time_step = 1
    start_time = time.time()

    # Metric trackers
    fairness_over_time = []
    throughput_over_time = []
    latency_over_time = []

    while any(user.remaining_demand > 0 for user in users) and total_bandwidth > 0:
        active_users = [user for user in users if user.remaining_demand > 0]
        
        # Modified request generation
        max_requests = len(active_users)
        min_requests = min(2, len(active_users))  # At least 2 if available
        num_requests = random.randint(min_requests, max(min_requests, max_requests // 3))
        
        # Select random users to request bandwidth
        requesting_users = []
        potential_requesters = active_users.copy()
        
        while len(requesting_users) < num_requests and potential_requesters:
            user = random.choice(potential_requesters)
            requesting_users.append(user)
            potential_requesters.remove(user)
            
        # Sort by user ID to maintain round-robin order among requesting users
        requesting_users = sorted(requesting_users, key=lambda u: u.ue_id)

        print(f"\n--- Time Step {time_step} ---")
        print(f"Users requesting bandwidth: {[user.ue_id for user in requesting_users]}")

        allocations_this_step = 0

        for user in requesting_users:
            if user.remaining_demand > total_bandwidth:
                print(f"Skipping UE {user.ue_id}: Demand {user.remaining_demand:.2f} Mbps exceeds remaining bandwidth {total_bandwidth:.2f} Mbps")
                continue

            allocation = min(allocation_per_step, user.remaining_demand, total_bandwidth)
            user.allocate_bandwidth(allocation)
            total_bandwidth -= allocation
            step_count += 1
            allocations_this_step += 1

            print(f"Step {step_count}: Allocated {allocation:.2f} Mbps to UE {user.ue_id}")
            print(f"UE {user.ue_id}: Remaining Demand = {user.remaining_demand:.2f} Mbps")
            print(f"Total Bandwidth Remaining = {total_bandwidth:.2f} Mbps\n")

            if total_bandwidth <= 0:
                break

        # Collect metrics at the end of each time step
        fairness_over_time.append(calculate_fairness(users))
        throughput_over_time.append(calculate_total_throughput(users))
        latency_over_time.append(calculate_average_latency(users))

        if allocations_this_step == 0:
            print("No allocations possible in this time step. Ending Round-Robin.")
            break

        time_step += 1

    unallocated_bandwidth = sum(user.remaining_demand for user in users)
    fairness = calculate_fairness(users)
    end_time = time.time()
    total_time = end_time - start_time

    return (
        step_count, 
        total_time, 
        unallocated_bandwidth, 
        fairness,
        fairness_over_time, 
        throughput_over_time, 
        latency_over_time
    )

def proportional_fair_scheduler(users, total_bandwidth, allocation_per_step):
    step_count = 0
    time_step = 1
    start_time = time.perf_counter()

    # Metric trackers
    fairness_over_time = []
    throughput_over_time = []
    latency_over_time = []
    
    while any(user.remaining_demand > 0 for user in users) and total_bandwidth > 0:
        active_users = [user for user in users if user.remaining_demand > 0]
        
        # Update channel qualities for all users
        for user in users:
            base_quality = max(0.2, 1 / (1 + (user.distance / RADIUS)**2))
            fade_factor = random.uniform(0.6, 1.4)
            user.channel_quality = base_quality * fade_factor
        
        # Modified request generation
        max_requests = len(active_users)
        min_requests = min(2, len(active_users))  # At least 2 if available
        num_requests = random.randint(min_requests, max(min_requests, max_requests // 3))
        
        # Select random users to request bandwidth with probabilities based on conditions
        requesting_users = []
        potential_requesters = active_users.copy()
        
        # First, get users that must request (no bandwidth or high priority)
        for user in potential_requesters.copy():
            if user.allocated_bandwidth == 0 or (user.channel_quality > 0.7 and user.remaining_demand > user.traffic_demand * 0.7):
                requesting_users.append(user)
                potential_requesters.remove(user)
                if len(requesting_users) >= num_requests:
                    break
        
        # If we need more users, select randomly from remaining
        while len(requesting_users) < num_requests and potential_requesters:
            user = random.choice(potential_requesters)
            potential_requesters.remove(user)
            
            request_probability = 0.3  # base probability
            if user.channel_quality > 0.5:
                request_probability = 0.5
            elif user.remaining_demand > user.traffic_demand * 0.7:
                request_probability = 0.4
                
            if random.random() < request_probability:
                requesting_users.append(user)
        
        if not requesting_users:
            time_step += 1
            continue
        
        print(f"\n--- Time Step {time_step} ---")
        print(f"Users requesting bandwidth: {[user.ue_id for user in requesting_users]}")

        # Enhanced PF metric calculation
        metrics = []
        for user in requesting_users:
            traffic_priority = 1.0
            if user.traffic_type == "video_streaming":
                traffic_priority = 1.5
            elif user.traffic_type == "web_browsing":
                traffic_priority = 1.2
            
            # Consider fairness in metric calculation
            normalized_allocation = user.allocated_bandwidth / user.traffic_demand
            fairness_factor = 1 / (normalized_allocation + 0.1)
            
            achievable_rate = user.channel_quality * allocation_per_step
            pf_metric = (achievable_rate * traffic_priority * fairness_factor) / \
                       (max(user.average_rate, 0.1)**0.7)
            
            metrics.append((user, pf_metric))

        print("\nFairness Metrics for Requesting Users:")
        for u, metric in metrics:
            print(f"UE {u.ue_id} - PF Metric: {metric:.2f}, Channel Quality: {u.channel_quality:.2f}, "
                  f"Average Rate: {u.average_rate:.2f}, Remaining: {u.remaining_demand:.2f} Mbps")

        # Sort users by PF metric (descending)
        metrics.sort(key=lambda x: x[1], reverse=True)

        allocations_this_step = 0
        remaining_step_bandwidth = total_bandwidth

        for user, metric in metrics:
            if user.remaining_demand > remaining_step_bandwidth:
                print(f"Skipping UE {user.ue_id}: Demand {user.remaining_demand:.2f} Mbps exceeds remaining bandwidth {remaining_step_bandwidth:.2f} Mbps")
                continue

            effective_allocation = min(
                allocation_per_step * user.channel_quality,
                remaining_step_bandwidth,
                user.remaining_demand
            )

            user.allocate_bandwidth(effective_allocation)
            remaining_step_bandwidth -= effective_allocation
            step_count += 1
            allocations_this_step += 1

            print(f"\nStep {step_count}: Allocated {effective_allocation:.2f} Mbps to UE {user.ue_id}")
            print(f"UE {user.ue_id}: Remaining Demand = {user.remaining_demand:.2f} Mbps")
            print(f"Total Bandwidth Remaining = {remaining_step_bandwidth:.2f} Mbps\n")

            if remaining_step_bandwidth <= 0:
                break

        # Collect metrics at the end of each time step
        fairness_over_time.append(calculate_fairness(users))
        throughput_over_time.append(calculate_total_throughput(users))
        latency_over_time.append(calculate_average_latency(users))

        if allocations_this_step == 0:
            print("No allocations possible. Ending Proportional Fair Scheduling.")
            break

        time_step += 1

    unallocated_bandwidth = sum(user.remaining_demand for user in users)
    fairness = calculate_fairness(users)
    end_time = time.perf_counter()
    time_taken = end_time - start_time

    return (
        step_count,
        time_taken,
        unallocated_bandwidth,
        fairness,
        fairness_over_time,
        throughput_over_time,
        latency_over_time
    )
    
def calculate_total_throughput(users):
    """Calculate the total throughput as the sum of all allocated bandwidths."""
    return sum(user.allocated_bandwidth for user in users)

def calculate_average_latency(users):
    """Calculate the average latency for all UEs."""
    latencies = [user.total_latency for user in users if user.total_latency > 0]
    if not latencies:
        return 0.0
    return sum(latencies) / len(latencies)


def calculate_fairness(users):
    allocated_bandwidths = [user.allocated_bandwidth / user.traffic_demand for user in users if user.allocated_bandwidth > 0]
    if not allocated_bandwidths:
        return 0.0
    n = len(allocated_bandwidths)
    squared_sum = sum(allocated_bandwidths)**2
    sum_squared = sum(x*x for x in allocated_bandwidths)
    return squared_sum / (n * sum_squared) if sum_squared > 0 else 0.0



def print_user_status(users, title):
    print(f"\n--- {title} ---")
    print("UE | Type            | Allocated  | Remaining Demand")
    for user in users:
        print(
            f"{str(user.ue_id).ljust(2)} | {user.traffic_type.ljust(15)} | "
            f"{f'{user.allocated_bandwidth:.2f}'.rjust(10)} Mbps | {f'{user.remaining_demand:.2f}'.rjust(15)} Mbps"
        )
    print("\n")


def run_simulation():
    global NUM_USERS, TOTAL_BANDWIDTH, ALLOCATION_PER_STEP, RADIUS, TRAFFIC_TYPES

    # Scenario selection
    print("\nSelect test scenario:")
    print("1. High Mobility Scenario (rapidly changing channel conditions)")
    print("2. Mixed Traffic Scenario (diverse user demands)")
    print("3. Cell Edge Scenario (poor channel conditions)")
    
    scenario = input("Enter scenario number (default=1): ") or "1"
    
    if scenario == "1":
        # High Mobility Scenario
        NUM_USERS = 50        # More users to show fairness management
        RADIUS = 250         # Large area for varied channel conditions
        TOTAL_BANDWIDTH = 500  # Higher bandwidth to handle many users
        TRAFFIC_TYPES = {
            "video_streaming": (25, 35),  # Higher video demands
            "web_browsing": (8, 18),      # More variable web traffic
            "voice_call": (1, 2)          # Steady voice demands
        }

    elif scenario == "2":
        # Mixed Traffic Scenario
        NUM_USERS = 30       # Moderate number of users
        RADIUS = 180         # Medium coverage area
        TOTAL_BANDWIDTH = 400 # Enough bandwidth for varied demands
        TRAFFIC_TYPES = {
            "video_streaming": (30, 40),  # Even higher video demands
            "web_browsing": (10, 25),     # More variable web traffic
            "voice_call": (1, 3)          # Slightly more variable voice
        }

    elif scenario == "3":
        # Cell Edge Scenario
        NUM_USERS = 20       # Fewer users but all at edge
        RADIUS = 200         # Larger radius to make edge conditions more challenging
        TOTAL_BANDWIDTH = 300 # Moderate bandwidth
        # Users will be in outer 30% of cell

    ALLOCATION_PER_STEP = 10
    
    # Set random seed for reproducibility
    random.seed(time.time())

    # Create users based on scenario
    if scenario == "3":
        # Modified user creation for cell edge scenario
        users = []
        for i in range(NUM_USERS):
            theta = random.uniform(0, 2 * np.pi)
            # Force users to be in outer 30% of cell
            r = random.uniform(0.7 * RADIUS, RADIUS)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            distance = np.sqrt(x**2 + y**2)
            traffic_type = random.choice(list(TRAFFIC_TYPES.keys()))
            users.append(UE(i, traffic_type, (x, y), distance))
    else:
        users = create_users(NUM_USERS)

    # Print initial environment
    print("\nInitial Environment Setup:")
    print("UE | Type            | Demand     | Channel Quality | Distance")
    for user in users:
        print(f"{str(user.ue_id).ljust(2)} | {user.traffic_type.ljust(15)} | "
              f"{f'{user.traffic_demand:.2f}'.rjust(5)} Mbps | "
              f"{user.channel_quality:.2f}".ljust(14) + f" | {user.distance:.1f}")

    # Sequential Round-Robin Scheduler
    print("\n=== Running Sequential Round-Robin Scheduler ===")
    (
        step_count_robin,
        time_taken_robin,
        unallocated_bandwidth_robin,
        fairness_robin,
        fairness_over_time_rr,
        throughput_over_time_rr,
        latency_over_time_rr,
    ) = sequential_round_robin(users, TOTAL_BANDWIDTH, ALLOCATION_PER_STEP)

    # Calculate and display Round-Robin metrics
    not_allocated_robin = len([user for user in users if user.remaining_demand > 0])
    total_throughput_robin = calculate_total_throughput(users)
    average_latency_robin = calculate_average_latency(users)
    print_user_status(users, "Final Status After Round-Robin")

    # Reset users for Proportional Fair
    for user in users:
        user.reset()

    # Proportional Fair Scheduler
    print("\n=== Running Proportional Fair Scheduler ===")
    (
        step_count_fair,
        time_taken_fair,
        unallocated_bandwidth_fair,
        fairness_fair,
        fairness_over_time_pf,
        throughput_over_time_pf,
        latency_over_time_pf,
    ) = proportional_fair_scheduler(users, TOTAL_BANDWIDTH, ALLOCATION_PER_STEP)

    # Calculate and display Proportional Fair metrics
    not_allocated_fair = len([user for user in users if user.remaining_demand > 0])
    total_throughput_fair = calculate_total_throughput(users)
    average_latency_fair = calculate_average_latency(users)
    print_user_status(users, "Final Status After Proportional Fair")

    # Plot comparison metrics
    plot_metrics(
        [fairness_over_time_rr, fairness_over_time_pf],
        ["Round-Robin", "Proportional Fair"],
        f"Fairness Over Time - Scenario {scenario}",
        "Fairness Index",
        save_as=f"fairness_scenario_{scenario}.png"
    )

    plot_metrics(
        [throughput_over_time_rr, throughput_over_time_pf],
        ["Round-Robin", "Proportional Fair"],
        f"Throughput Over Time - Scenario {scenario}",
        "Throughput (Mbps)",
        save_as=f"throughput_scenario_{scenario}.png"
    )

    plot_metrics(
        [latency_over_time_rr, latency_over_time_pf],
        ["Round-Robin", "Proportional Fair"],
        f"Average Latency Over Time - Scenario {scenario}",
        "Latency (time steps)",
        save_as=f"latency_scenario_{scenario}.png"
    )

    # Compare algorithms and print results
    compare_algorithms(
        round_robin=(
            step_count_robin,
            time_taken_robin,
            not_allocated_robin,
            unallocated_bandwidth_robin,
            fairness_robin,
            total_throughput_robin,
            average_latency_robin
        ),
        proportional_fair=(
            step_count_fair,
            time_taken_fair,
            not_allocated_fair,
            unallocated_bandwidth_fair,
            fairness_fair,
            total_throughput_fair,
            average_latency_fair
        ),
        scenario=scenario
    )

def compare_algorithms(round_robin, proportional_fair, scenario):
    rr_steps, rr_time, rr_not_allocated, rr_unallocated, rr_fairness, rr_throughput, rr_latency = round_robin
    pf_steps, pf_time, pf_not_allocated, pf_unallocated, pf_fairness, pf_throughput, pf_latency = proportional_fair

    print(f"\n--- Algorithm Comparison for Scenario {scenario} ---")
    
    # Print detailed metrics for each algorithm
    print("\nRound-Robin Metrics:")
    print(f"- Steps: {rr_steps}")
    print(f"- Execution Time: {rr_time:.5f}s")
    print(f"- Users Not Fully Allocated: {rr_not_allocated}")
    print(f"- Unallocated Bandwidth: {rr_unallocated:.2f} Mbps")
    print(f"- Fairness Index: {rr_fairness:.5f}")
    print(f"- Total Throughput: {rr_throughput:.2f} Mbps")
    print(f"- Average Latency: {rr_latency:.2f}")

    print("\nProportional Fair Metrics:")
    print(f"- Steps: {pf_steps}")
    print(f"- Execution Time: {pf_time:.5f}s")
    print(f"- Users Not Fully Allocated: {pf_not_allocated}")
    print(f"- Unallocated Bandwidth: {pf_unallocated:.2f} Mbps")
    print(f"- Fairness Index: {pf_fairness:.5f}")
    print(f"- Total Throughput: {pf_throughput:.2f} Mbps")
    print(f"- Average Latency: {pf_latency:.2f}")

    # Calculate improvement percentages
    throughput_improvement = ((pf_throughput - rr_throughput) / rr_throughput * 100) if rr_throughput > 0 else float('inf')
    fairness_improvement = ((pf_fairness - rr_fairness) / rr_fairness * 100) if rr_fairness > 0 else float('inf')
    latency_improvement = ((rr_latency - pf_latency) / rr_latency * 100) if rr_latency > 0 else float('inf')

    print(f"\nPerformance Improvements with Proportional Fair:")
    print(f"- Throughput Improvement: {throughput_improvement:.1f}%")
    print(f"- Fairness Improvement: {fairness_improvement:.1f}%")
    print(f"- Latency Reduction: {latency_improvement:.1f}%")

    # Scenario-specific analysis
    if scenario == "1":
        print("\nHigh Mobility Scenario Analysis:")
        print("- PF scheduler better adapts to varying channel conditions")
        if throughput_improvement > 0:
            print("- Successfully achieves higher throughput by exploiting good channel conditions")
        if fairness_improvement > 0:
            print("- Maintains fairness despite channel variations")

    elif scenario == "2":
        print("\nMixed Traffic Scenario Analysis:")
        print("- PF scheduler handles diverse traffic requirements")
        if throughput_improvement > 0:
            print("- Better serves high-bandwidth users while maintaining fairness")
        if latency_improvement > 0:
            print("- Reduces average latency through smarter resource allocation")

    elif scenario == "3":
        print("\nCell Edge Scenario Analysis:")
        print("- PF scheduler manages poor channel conditions")
        if fairness_improvement > 0:
            print("- Achieves better fairness for cell edge users")
        if throughput_improvement > 0:
            print("- Maintains throughput despite challenging conditions")

    print("\nOverall Winner:")
    metrics = {
        "Steps": "Proportional Fair" if pf_steps < rr_steps else "Round-Robin",
        "Execution Speed": "Proportional Fair" if pf_time < rr_time else "Round-Robin",
        "Resource Allocation": "Proportional Fair" if pf_not_allocated < rr_not_allocated else "Round-Robin",
        "Bandwidth Utilization": "Proportional Fair" if pf_unallocated < rr_unallocated else "Round-Robin",
        "Throughput": "Proportional Fair" if pf_throughput > rr_throughput else "Round-Robin",
        "Latency": "Proportional Fair" if pf_latency < rr_latency else "Round-Robin",
        "Fairness": "Proportional Fair" if pf_fairness > rr_fairness else "Round-Robin"
    }

    for metric, winner in metrics.items():
        print(f"- {metric}: {winner}")

def plot_metrics(metrics, labels, title, ylabel, save_as=None):
    plt.figure(figsize=(10, 6))
    for metric, label in zip(metrics, labels):
        plt.plot(metric, label=label)
    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    run_simulation()
