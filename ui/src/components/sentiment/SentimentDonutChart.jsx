import ReactECharts from "echarts-for-react";

const SentimentDonutChart = ({ N_pos, N_neg }) => {
  const total = N_pos + N_neg;

  const option = {
    tooltip: {
      trigger: "item",
      formatter: "{b}: {c}건 ({d}%)"
    },
    legend: {
      bottom: 0,
      orient: "vertical",
      left: "left",
      icon: "circle",
      data: ["긍정 뉴스", "부정 뉴스"]
    },
    series: [
      {
        name: "감성 분포",
        type: "pie",
        radius: ["50%", "80%"],
        avoidLabelOverlap: false,
        label: {
          show: true,
          position: "outside",
          formatter: "{b}\n{d}%"
        },
        labelLine: {
          show: true
        },
        data: [
          { value: N_pos, name: "긍정 뉴스", itemStyle: { color: "#3b82f6" } },
          { value: N_neg, name: "부정 뉴스", itemStyle: { color: "#ef4444" } }
        ]
      }
    ],
    graphic: {
      elements: [
        {
          type: "text",
          left: "center",
          top: "center",
          style: {
            text: `${total}건`,
            fontSize: 18,
            fontWeight: 600,
            fill: "#111827"
          }
        }
      ]
    }
  };

  return (
    <div
      style={{
        marginTop: "-15px",
        backgroundColor: "#fff",
        border: "1px solid #e5e7eb",
        borderRadius: "8px",
        padding: "12px",
        boxShadow: "0 2px 4px rgba(0,0,0,0.03)",
        height: "48%",
        minHeight: "260px"
      }}
    >
      <h3 style={{ textAlign: "center", fontSize: "16px", marginTop: "0px", marginBottom: "40px" }}>
        감성 분포 (도넛)
      </h3>
      <ReactECharts option={option} style={{ width: "100%", height: "200px" }} />
    </div>
  );
};

export default SentimentDonutChart;
