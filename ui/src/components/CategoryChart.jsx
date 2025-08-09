import ReactECharts from "echarts-for-react";
import cpiData from "../data/parsedCPIData_with_full_predictions.json";

const CategoryChart = ({ selectedCategory, startDate, endDate }) => {
  const records = selectedCategory ? cpiData[selectedCategory] : [];

  const allDates = records
    .map((r) => r.날짜)
    .filter((d) => d >= startDate && d <= endDate);

  const actual = allDates.map((date) => {
    const item = records.find((r) => r.날짜 === date);
    return item?.값 ?? null;
  });

  const predicted = allDates.map((date) => {
    const item = records.find((r) => r.날짜 === date);
    return item?.예측값 ?? null;
  });

  const actualColor = "#3b82f6";
  const predictedColor = "#ef4444";

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
        name: `${selectedCategory || "항목"} (실제)`,
        type: "line",
        data: actual,
        smooth: true,
        symbol: "circle",
        symbolSize: 6,
        lineStyle: { width: 2 },
        itemStyle: { color: actualColor },
      },
      {
        name: `${selectedCategory || "항목"} (예측)`,
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
          color: predictedColor,
        },
      },
    ],
  };

  const fallbackOption = {
    title: {
      text: "항목을 선택해주세요",
      left: "center",
      top: "middle",
      textStyle: {
        fontSize: 14,
        color: "#9ca3af",
      },
    },
    xAxis: {
      type: "category",
      data: [],
      axisLine: { lineStyle: { color: "#d1d5db" } },
    },
    yAxis: {
      type: "value",
      axisLine: { lineStyle: { color: "#d1d5db" } },
    },
    series: [],
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
        {selectedCategory || "품목"} 추세 차트
      </h3>

      <ReactECharts
        key={selectedCategory || "empty"}
        option={
          selectedCategory ? option : fallbackOption
        }
        notMerge={true}
        style={{ width: "100%", height: 280 }}
      />
    </div>
  );
};

export default CategoryChart;
