import numpy as np
import random
import matplotlib.pyplot as plt
import copy


BETA = 0.5  # Weight for averaging past and current rates
NUM_USERS = 200
TOTAL_BANDWIDTH = 300
CHECKPOINT_INTERVAL = 10

class UE:
    def __init__(self, ue_id, traffic_type):
        self.ue_id = ue_id
        self.traffic_type = traffic_type
        self.channel_quality = self.calculate_initial_channel_quality()
        self.traffic_demand = self.generate_traffic_demand(traffic_type)
        self.remaining_demand =  2*self.traffic_demand
        self.initial_total_demand = self.remaining_demand
        self.total_latency = 0
        self.instantaneous_rate = 0
        self.average_rate = 0
        self.active = False

    def calculate_initial_channel_quality(self):
        return random.uniform(0.2, 1.0)  # Random channel quality between 0.2 and 1.0

    def generate_traffic_demand(self, traffic_type):
        min_demand, max_demand = TRAFFIC_TYPES[traffic_type]
        return random.uniform(min_demand, max_demand)

    def allocate_bandwidth(self, amount):
        if self.active:
            allocation = min(amount, self.remaining_demand)
            self.remaining_demand -= allocation
            self.instantaneous_rate = allocation
            self.update_average_rate()
            if self.remaining_demand <= 0:
                self.active = False

    def update_average_rate(self):
        self.average_rate = (1 - BETA) * self.average_rate + BETA * self.instantaneous_rate

    def increment_latency(self):
        if self.active and self.remaining_demand > 0:
            self.total_latency += 1

    def reset(self):
        self.remaining_demand = self.initial_total_demand
        self.total_latency = 0
        self.instantaneous_rate = 0
        self.average_rate = 0
        self.active = False


def create_users(num_users):
    users = []
    for i in range(num_users):
        traffic_type = random.choice(list(TRAFFIC_TYPES.keys()))
        users.append(UE(i, traffic_type))
    return users

def sequential_round_robin(users, total_bandwidth):
    fairness_over_time = []
    throughput_over_time = []
    latency_over_time = []

    for step in range(CHECKPOINT_INTERVAL):
        # Ensure valid users have remaining demand
        valid_users = [user for user in users if user.remaining_demand > 0]
        if not valid_users:
            break  # Stop if no users have remaining demand
        
        if len(valid_users) < 20:
            max_active_users = len(valid_users)
        else:
            max_active_users = random.randint(10, len(valid_users) // 2)

        active_users_list = random.sample(valid_users, max_active_users)
        
        for user in users:
            user.active = user in active_users_list
            if user.active:
                user.increment_latency()

        active_users = [user for user in users if user.active]

        dynamic_bandwidth = total_bandwidth
        bandwidth_per_user = dynamic_bandwidth / len(active_users) if active_users else 0

        print(f"\n{'='*40}")
        print(f"Time Step {step + 1}")
        print(f"Active Users: {len(active_users)}")
        print(f"Dynamic Bandwidth: {dynamic_bandwidth:.2f} Mbps")
        print(f"Bandwidth per User: {bandwidth_per_user:.2f} Mbps")
        print(f"{'='*40}")

        if active_users:
            print(f"{'User Allocation Details (Before Allocation)':^80}")
            print(f"{'User ID':<10}{'Initial Demand (Mbps)':<25}{'Remaining Demand (Mbps)':<20}{'Channel Quality':<20}")
            print("-" * 80)
            for user in active_users:
                print(f"{user.ue_id:<10}{user.initial_total_demand:<25.2f}{user.remaining_demand:<25.2f}{user.channel_quality:<20.2f}")
            print()
            for idx, user in enumerate(active_users):
                print(f"Step {idx + 1}: ID {user.ue_id:<3} | "
                      f"Initial Demand: {user.initial_total_demand:.2f} | "
                      f"Channel Quality: {user.channel_quality:.2f} | "
                      f"Allocated Bandwidth: {bandwidth_per_user:.2f} | "
                      f"Remaining Demand Before: {user.remaining_demand:.2f}")
                    
                
                user.allocate_bandwidth(bandwidth_per_user*user.channel_quality)
                dynamic_bandwidth -= bandwidth_per_user  # Update the remaining bandwidth
                print(f"    Remaining Demand After: {user.remaining_demand:.2f}")
                print(f"    Base Station Remaining Bandwidth After Allocation: {dynamic_bandwidth:.2f}")
            print("-" * 80)

        # Compute metrics
        total_throughput = sum(user.instantaneous_rate for user in active_users)
        fairness = calculate_fairness(active_users)
        avg_latency = calculate_average_latency(users)

        throughput_over_time.append(total_throughput)
        fairness_over_time.append(fairness)
        latency_over_time.append(avg_latency)

    return fairness_over_time, throughput_over_time, latency_over_time

def proportional_fair_scheduler(users, total_bandwidth):
    fairness_over_time = []
    throughput_over_time = []
    latency_over_time = []

    for step in range(CHECKPOINT_INTERVAL):
        # Ensure valid users have remaining demand
        valid_users = [user for user in users if user.remaining_demand > 0]
        if not valid_users:
            break  # Stop if no users have remaining demand

        if len(valid_users) < 20:
            max_active_users = len(valid_users)
        else:
            max_active_users = random.randint(10, len(valid_users) // 2)
        active_users_list = random.sample(valid_users, max_active_users)

        for user in users:
            user.active = user in active_users_list
            if user.active:
                user.increment_latency()

        active_users = [user for user in users if user.active]

        dynamic_bandwidth = total_bandwidth
        remaining_bandwidth = dynamic_bandwidth

        print(f"\n{'='*40}")
        print(f"Time Step {step + 1}")
        print(f"Active Users: {len(active_users)}")
        print(f"Dynamic Bandwidth: {dynamic_bandwidth:.2f} Mbps")
        print(f"{'='*40}")

        if active_users:
            print(f"{'User Allocation Details (Before Allocation)':^80}")
            print(f"{'User ID':<10}{'Initial Demand (Mbps)':<25}{'Remaining Demand (Mbps)':<25}{'Channel Quality':<20}{'PF Metric':<20}")
            print("-" * 80)

            metrics = [
                (user, user.channel_quality / max(user.average_rate, 0.1)) for user in active_users
            ]
            metrics.sort(key=lambda x: x[1], reverse=True)

            for user, metric in metrics:
                print(f"{user.ue_id:<10}{user.initial_total_demand:<25.2f}{user.remaining_demand:<25.2f}{user.channel_quality:<20.2f}{metric:<20.2f}")

            print(f"\n{'Step-by-Step Allocation':^80}")
            for idx, (user, metric) in enumerate(metrics):
                if remaining_bandwidth <= 0:
                    break
                bandwidth_allocation = min(user.remaining_demand / user.channel_quality, remaining_bandwidth)
                effective_allocation = bandwidth_allocation * user.channel_quality
                print(f"Step {idx + 1}: ID {user.ue_id:<3} | "
                      f"Initial Bandwidth: {user.initial_total_demand:.2f} | "
                      f"PF Metric: {metric:.2f} | "
                      f"Allocated Bandwidth: {effective_allocation:.2f} | "
                      f"Remaining Demand Before Allocation: {user.remaining_demand:.2f}")
                user.allocate_bandwidth(effective_allocation)
                remaining_bandwidth -= bandwidth_allocation
                print(f"Remaining Demand After Allocation: {user.remaining_demand:.2f}")
                print(f"    Base Station Remaining Bandwidth After Allocation: {remaining_bandwidth:.2f}")
            print("-" * 80)

        # Compute metrics
        total_throughput = sum(user.instantaneous_rate for user in active_users)
        fairness = calculate_fairness(active_users)
        avg_latency = calculate_average_latency(users)

        throughput_over_time.append(total_throughput)
        fairness_over_time.append(fairness)
        latency_over_time.append(avg_latency)

    return fairness_over_time, throughput_over_time, latency_over_time


def calculate_fairness(users):
    allocated_bandwidths = [user.instantaneous_rate for user in users if user.active]
    if not allocated_bandwidths:
        return 0
    squared_sum = sum(allocated_bandwidths)**2
    sum_squared = sum(x**2 for x in allocated_bandwidths)
    n = len(allocated_bandwidths)
    return squared_sum / (n * sum_squared) if sum_squared > 0 else 0


def calculate_average_latency(users):
    latencies = [user.total_latency for user in users if user.active or user.total_latency > 0]
    return sum(latencies) / len(latencies) if latencies else 0


def plot_metrics(metrics, labels, title, ylabel):
    plt.figure(figsize=(10, 6))
    for metric, label in zip(metrics, labels):
        plt.plot(metric, label=label)
    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()


def run_simulation():
    global NUM_USERS, TOTAL_BANDWIDTH, TRAFFIC_TYPES

    TRAFFIC_TYPES = {
        "video_streaming": (10, 25),
        "voice_call": (5, 10),
        "web_browsing": (1, 5)
    }

    users = create_users(NUM_USERS)
    users_rr = copy.deepcopy(users)
    users_pf = copy.deepcopy(users)

    print("\n=== Round Robin Scheduler ===")
    fairness_rr, throughput_rr, latency_rr = sequential_round_robin(users_rr, TOTAL_BANDWIDTH)

    print("\n=== Proportional Fair Scheduler ===")
    fairness_pf, throughput_pf, latency_pf = proportional_fair_scheduler(users_pf, TOTAL_BANDWIDTH)

    plot_metrics(
        [fairness_rr, fairness_pf],
        ["Round Robin", "Proportional Fair"],
        "Fairness Over Time",
        "Fairness Index"
    )
    plot_metrics(
        [throughput_rr, throughput_pf],
        ["Round Robin", "Proportional Fair"],
        "Throughput Over Time",
        "Throughput (Mbps)"
    )
    plot_metrics(
        [latency_rr, latency_pf],
        ["Round Robin", "Proportional Fair"],
        "Latency Over Time",
        "Latency (time steps)"
    )


if __name__ == "__main__":
    random.seed(10)
    np.random.seed(10)
    run_simulation()
