TEMPLATE = "vless://40e009a2-458a-45be-88ad-c3f60fac0dab@{ip}:{port}?encryption=none&security=tls&sni=liuz.ccwu.cc&fp=chrome&insecure=0&allowInsecure=0&type=ws&host=liuz.ccwu.cc&path=%2F#{tag}"

INPUT_FILE = "data/20260619.txt"
OUTPUT_FILE = "output/output_vless.txt"

def parse_line(line):
    line = line.strip()
    if not line or line.startswith("#") or not line.endswith("SG"):
        return None

    # 101.99.76.88:2053#NL
    if "#" in line :
        ip_port, tag = line.split("#")
    else:
        ip_port, tag = line, "US"

    ip, port = ip_port.split(":")
    return ip, port, tag


def main():
    results = []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            parsed = parse_line(line)
            if not parsed:
                continue

            ip, port, tag = parsed

            vless = TEMPLATE.format(
                ip=ip.strip(),
                port=port.strip(),
                tag=tag.strip()
            )

            results.append(vless)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in results:
            f.write(r + "\n")

    print(f"完成！生成 {len(results)} 条 VLESS 节点")
    print(f"输出文件：{OUTPUT_FILE}")


if __name__ == "__main__":
    main()