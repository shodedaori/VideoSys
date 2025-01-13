import json

num_samples_per_type = 16


def get_small_vbench():
    full_path = "VBench_full_info.json"

    with open(full_path, "r") as f:
        full_info = json.load(f)

    small_bench = []
    sample_count = None
    
    total_count = 0
    prev_type = None
    
    for test in full_info:
        this_type = test["dimension"]
        if prev_type != this_type:
            sample_count = num_samples_per_type
            total_count += 1
        
        if sample_count > 0:
            sample_count -= 1
            small_bench.append(test)

        prev_type = this_type
    
    print(f"The full vbench has {total_count} types")
    print(f"Writing {len(small_bench)} samples to small_vbench.json")

    with open("middle_vbench.json", "w") as f:
        json.dump(small_bench, f, indent=4)


if __name__ == "__main__":
    get_small_vbench()
