import ReactECharts from "echarts-for-react";

const SentimentTrendLine = ({ months, sentiment, cpiActual, cpiPredicted }) => {
  // 감성 점수 범위 계산
  const minSent = Math.min(...sentiment);
  const maxSent = Math.max(...sentiment);
  const sentimentMin = Math.floor(minSent * 10) / 10 - 0.05;
  const sentimentMax = Math.ceil(maxSent * 10) / 10 + 0.05;

  // CPI 범위 계산
  const allCpi = [...cpiActual, ...cpiPredicted].filter((v) => typeof v === "number");
  const minCpi = Math.floor(Math.min(...allCpi));
  const maxCpi = Math.ceil(Math.max(...allCpi));
  const cpiMin = minCpi - 1;
  const cpiMax = maxCpi + 1;

  const option = {
    tooltip: { trigger: "axis" },
    legend: { top: 10, left: "center" },
    grid: {
      left: 40,
      right: 40,
      top: 60,
      bottom: 60
    },
    xAxis: {
      type: "category",
      data: months,
      axisLabel: {
        rotate: 45,
        margin: 12
      }
    },
    yAxis: [
      {
        type: "value",
        name: "CPI",
        position: "left",
        min: cpiMin,
        max: cpiMax,
        axisLabel: { formatter: "{value}" }
      },
      {
        type: "value",
        name: "감성 점수",
        position: "right",
        min: sentimentMin,
        max: sentimentMax,
        axisLabel: { formatter: "{value}" }
      }
    ],
    series: [
      {
        name: "CPI (실제)",
        type: "line",
        yAxisIndex: 0,
        data: cpiActual,
        smooth: true,
        symbol: "circle",
        lineStyle: { width: 2, color: "#2563eb" },
        itemStyle: { color: "#2563eb" }
      },
      {
        name: "CPI (예측)",
        type: "line",
        yAxisIndex: 0,
        data: cpiPredicted,
        smooth: true,
        lineStyle: { type: "dashed", width: 2, color: "#ef4444" },
        itemStyle: { color: "#ef4444" }
      },
      {
        name: "감성 점수",
        type: "line",
        yAxisIndex: 1,
        data: sentiment,
        smooth: true,
        symbol: "circle",
        symbolSize: 6,
        lineStyle: { width: 3, color: "#10b981" },
        itemStyle: { color: "#10b981" }
      }
    ]
  };

  return (
    <div
      style={{
        backgroundColor: "#fff",
        border: "1px solid #e5e7eb",
        borderRadius: "8px",
        padding: "12px",
        boxShadow: "0 2px 4px rgba(0,0,0,0.03)",
        height: "48%",
        minHeight: "260px"
      }}
    >
      <h3 style={{ textAlign: "center", fontSize: "16px", marginBottom: "8px" }}>
        감성 점수 vs CPI 추세
      </h3>
      <ReactECharts option={option} style={{ width: "100%", height: "90%" }} />
    </div>
  );
};

export default SentimentTrendLine;
