import random
import time
import argparse

# Define parameters
NUM_USERS = 10  # Number of User Equipments (UEs)
TOTAL_BANDWIDTH = 50  # in Mbps, total bandwidth available at the base station
DISPATCH_INTERVAL = 1  # seconds, interval to simulate sending records
ALLOCATION_PER_STEP = 10  # in Mbps, fixed amount allocated per step

# Traffic Types
TRAFFIC_TYPES = {
    "video_streaming": (15, 20),  # Continuous high bandwidth demand
    "web_browsing": (1, 10),  # Sporadic bursts
    "voice_call": (1, 3)  # Constant low bandwidth demand
}

# Define User Equipment (UE) class
class UE:
    def __init__(self, ue_id, traffic_type):
        self.ue_id = ue_id
        self.traffic_type = traffic_type
        self.traffic_demand = self.generate_traffic_demand(traffic_type)  # Mbps
        self.channel_quality = random.uniform(0.5, 1.5)  # Initial channel quality
        self.allocated_bandwidth = 0
        self.remaining_demand = self.traffic_demand
        self.total_latency = 0  # Track latency in terms of how many intervals it took to meet demand

    def generate_traffic_demand(self, traffic_type):
        min_demand, max_demand = TRAFFIC_TYPES[traffic_type]
        return random.uniform(min_demand, max_demand)

    def vary_channel_quality(self):
        # Simulate random fluctuations in channel quality
        self.channel_quality = max(0.1, min(2.0, self.channel_quality + random.uniform(-0.2, 0.2)))

    def allocate_bandwidth(self, amount):
        # Allocate bandwidth up to the remaining demand
        allocation = min(amount, self.remaining_demand)
        self.allocated_bandwidth += allocation
        self.remaining_demand -= allocation
        if allocation > 0:
            self.total_latency += 1  # Track time it took to allocate resources

    def get_fairness_metric(self):
        # Avoid division by zero by setting a minimum throughput value
        average_throughput = max(0.1, self.allocated_bandwidth)
        return self.channel_quality / average_throughput

    def __repr__(self):
        return (f"UE {self.ue_id} | Type: {self.traffic_type} | "
                f"Demand: {self.traffic_demand:.2f} Mbps | "
                f"Allocated: {self.allocated_bandwidth:.2f} Mbps | "
                f"Remaining: {self.remaining_demand:.2f} Mbps")

# Initialize UEs with different traffic types
def create_users(num_users):
    traffic_patterns = ["video_streaming", "web_browsing", "voice_call"]
    users = []
    for i in range(num_users):
        traffic_type = random.choice(traffic_patterns)
        users.append(UE(i, traffic_type))
    return users

# Proportional Fair Scheduling Algorithm with Fairness Metric
def proportional_fair_scheduler(users, total_bandwidth):
    available_bandwidth = total_bandwidth
    step_count = 0
    start_time = time.time()
    
    while available_bandwidth > 0 and any(user.remaining_demand > 0 for user in users):
        # Update channel qualities to simulate real-time changes
        for user in users:
            user.vary_channel_quality()

        # Calculate the fairness metric for each user
        metrics = [(user, user.get_fairness_metric()) for user in users if user.remaining_demand > 0]
        if not metrics:
            break

        # Print all fairness metrics for this step, including demand and allocations
        print(f"\nStep {step_count + 1}: Fairness Metrics:")
        for u, metric in metrics:
            print(f"UE {u.ue_id} - Fairness Metric: {metric:.2f}, Channel Quality: {u.channel_quality:.2f}, "
                  f"Demand: {u.traffic_demand:.2f} Mbps, Allocated: {u.allocated_bandwidth:.2f} Mbps, "
                  f"Remaining: {u.remaining_demand:.2f} Mbps")

        # Select the user with the highest fairness metric
        selected_user, selected_metric = max(metrics, key=lambda x: x[1])
        
        # Allocate a fixed step of 10 Mbps or up to the remaining demand
        allocation = min(ALLOCATION_PER_STEP, available_bandwidth, selected_user.remaining_demand)
        selected_user.allocate_bandwidth(allocation)
        available_bandwidth -= allocation
        step_count += 1

        print(f"\nStep {step_count}: Allocated {allocation:.2f} Mbps to UE {selected_user.ue_id} based on Fairness Metric {selected_metric:.2f}")
        print(f"Remaining Bandwidth: {available_bandwidth:.2f} Mbps")
    
    end_time = time.time()
    total_duration = end_time - start_time
    return step_count, available_bandwidth, total_duration

# Sequential Round-Robin Allocation
def sequential_round_robin(users, total_bandwidth, allocation_per_step):
    step_count = 0
    start_time = time.perf_counter()  # Use high-resolution timer
    
    while any(user.remaining_demand > 0 for user in users) and total_bandwidth > 0:
        for user in users:
            if user.remaining_demand > 0:
                # Determine the allocation amount as the minimum of the remaining demand, allocation step, and available bandwidth
                allocation = min(allocation_per_step, user.remaining_demand, total_bandwidth)
                
                user.allocate_bandwidth(allocation)
                total_bandwidth -= allocation
                step_count += 1
                
                # Print detailed information for each step
                print(f"\n--- Step {step_count}: Allocated {allocation:.2f} Mbps to UE {user.ue_id} ---")
                print(f"UE {user.ue_id} - Demand: {user.traffic_demand:.2f} Mbps, "
                      f"Allocated: {user.allocated_bandwidth:.2f} Mbps, "
                      f"Remaining: {user.remaining_demand:.2f} Mbps")
                print(f"Remaining Total Bandwidth: {max(total_bandwidth, 0):.2f} Mbps")
                
                # Exit if no bandwidth remains
                if total_bandwidth <= 0:
                    end_time = time.perf_counter()  # Use high-resolution timer for end time
                    total_duration = end_time - start_time
                    return step_count, max(total_bandwidth, 0), total_duration
    
    end_time = time.perf_counter()  # Use high-resolution timer for end time
    total_duration = end_time - start_time
    return step_count, max(total_bandwidth, 0), total_duration


# Calculate Metrics
def calculate_metrics(users):
    # Total throughput: Sum of all allocated bandwidths
    total_throughput = sum(user.allocated_bandwidth for user in users)

    # Fairness Index (Jain's Index)
    allocated_list = [user.allocated_bandwidth for user in users]
    fairness_index = (sum(allocated_list) ** 2) / (NUM_USERS * sum(x ** 2 for x in allocated_list)) if sum(allocated_list) > 0 else 0

    # Average Latency: How many intervals on average it took for users to be served
    average_latency = sum(user.total_latency for user in users) / NUM_USERS

    return total_throughput, fairness_index, average_latency

# Simulate scheduling with timing and selection
def run_simulation(mode="proportional_fair"):
    # Create a fresh set of users
    users = create_users(NUM_USERS)

    # Display the initial environment
    print("\nInitial Environment (Before Allocation):")
    for user in users:
        print(f"UE {user.ue_id} | Type: {user.traffic_type} | Demand: {user.traffic_demand:.2f} Mbps | Channel Quality: {user.channel_quality:.2f}")

    if mode == "proportional_fair":
        print("\n--- Proportional Fair Scheduling ---")
        step_count, remaining_bandwidth, total_duration = proportional_fair_scheduler(users, TOTAL_BANDWIDTH)

        # Display Final Results
        print("\nFinal Allocations:")
        for user in users:
            print(f"UE {user.ue_id} | Type: {user.traffic_type} | Demand: {user.traffic_demand:.2f} Mbps | "
                  f"Allocated: {user.allocated_bandwidth:.2f} Mbps | Remaining: {user.remaining_demand:.2f} Mbps")

        # Calculate and display metrics
        total_throughput, fairness_index, average_latency = calculate_metrics(users)
        print(f"\nTotal Throughput: {total_throughput:.2f} Mbps")
        print(f"Fairness Index: {fairness_index:.2f}")
        print(f"Average Latency: {average_latency:.2f} intervals")
        print(f"Total Steps Taken: {step_count}")
        print(f"Total Time Elapsed: {total_duration:.5f} seconds")
        print(f"Remaining Bandwidth: {remaining_bandwidth:.2f} Mbps")

    elif mode == "sequential_round_robin":
        print("\n--- Sequential Round-Robin Scheduling ---")
        step_count, remaining_bandwidth, total_duration = sequential_round_robin(users, TOTAL_BANDWIDTH, ALLOCATION_PER_STEP)

        # Display Final Results
        print("\nFinal Allocations:")
        for user in users:
            print(f"UE {user.ue_id} | Type: {user.traffic_type} | Demand: {user.traffic_demand:.2f} Mbps | "
                  f"Allocated: {user.allocated_bandwidth:.2f} Mbps | Remaining: {user.remaining_demand:.2f} Mbps")

        # Calculate and display metrics
        total_throughput, fairness_index, average_latency = calculate_metrics(users)
        print(f"\nTotal Throughput: {total_throughput:.2f} Mbps")
        print(f"Average Latency: {average_latency:.2f} intervals")
        print(f"Total Steps Taken: {step_count}")
        print(f"Total Time Elapsed: {total_duration:.5f} seconds")
        print(f"Remaining Bandwidth: {remaining_bandwidth:.2f} Mbps")

# Main entry point to run simulation based on command-line argument
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run network scheduling simulation.")
    parser.add_argument("mode", choices=["robin", "fair"], help="Choose scheduling mode: 'robin' for Round-Robin or 'fair' for Proportional Fair")
    args = parser.parse_args()

    if args.mode == "robin":
        run_simulation("sequential_round_robin")
    elif args.mode == "fair":
        run_simulation("proportional_fair")