import ReactECharts from "echarts-for-react";
import cpiData from "../data/parsedCPIData_with_full_predictions.json";

const CPIOnlyChart = ({ startDate, endDate }) => {
  const records = cpiData["CPI"];

  const allDates = records
    .map((r) => r.날짜)
    .filter((d) => d >= startDate && d <= endDate);

  const actual = allDates.map((date) =>
    records.find((r) => r.날짜 === date)?.값 ?? null
  );
  const predicted = allDates.map((date) =>
    records.find((r) => r.날짜 === date)?.예측값 ?? null
  );

  // 크래시 방지
  if (actual.every(v => v === null) && predicted.every(v => v === null)) {
    return (
      <div style={{ padding: 16, textAlign: "center", color: "#6b7280" }}>
        선택한 기간에 표시할 수 있는 데이터가 없습니다.
      </div>
    );
  }

  const actualColor = "#3b82f6";     // 파란색
  const predictedColor = "#ef4444"; // 빨간색

  const option = {
    tooltip: { trigger: "axis" },
    legend: { top: 10, left: "center" },
    grid: { left: 40, right: 20, top: 80, bottom: 60 },
    xAxis: {
      type: "category",
      data: allDates,
      axisLabel: { rotate: 45 },
    },
    yAxis: { type: "value" },
    series: [
      {
        name: "CPI (실제)",
        type: "line",
        data: actual,
        smooth: true,
        symbol: "circle",
        symbolSize: 6,
        lineStyle: { width: 2 },
        itemStyle: { color: actualColor },
      },
      {
        name: "CPI (예측)",
        type: "line",
        data: predicted,
        smooth: true,
        symbol: "none",
        lineStyle: {
          type: "dashed",
          width: 2,
          color: predictedColor,
        },
        itemStyle: {
          color: predictedColor, // <-- 이게 범례 색 결정
          
        },
      },
    ],
  };

  return (
    <div
      style={{
        backgroundColor: "white",
        border: "1px solid #e5e7eb",
        borderRadius: "8px",
        boxShadow: "0 2px 6px rgba(0,0,0,0.05)",
        padding: "5px",
        marginTop: "2.5px",
        marginBottom: "2.5px",
        minHeight: "400px",
        display: "flex",
        flexDirection: "column",
      }}
    >
      <h3
        style={{
          fontSize: "18px",
          fontWeight: "600",
          marginBottom: "16px",
          color: "#111827",
          textAlign: "center",
        }}
      >
        CPI 추세 차트
      </h3>

      <ReactECharts
        option={option}
        notMerge={true}
        style={{ width: "100%", height: 280 }}
      />
    </div>
  );
};

export default CPIOnlyChart;
