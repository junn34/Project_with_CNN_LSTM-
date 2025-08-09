import React, { useState, useEffect } from "react";
import CategoryFilter from "./components/CategoryFilter";
import DateRangePicker from "./components/DateRangePicker";
import CPIOnlyChart from "./components/CPIOnlyChart";
import CategoryChart from "./components/CategoryChart";
import CPITableCPI from "./components/CPITableCPI";
import CPITableCategory from "./components/CPITableCategory";
import KpiCard from "./components/KpiCard";

import cpiData from "./data/parsedCPIData_with_full_predictions.json";
import sentimentData from "./data/sentimentData.json";
import { getSentimentDashboardData } from "./utils/getSentimentDashboardData";

import MonthSelector from "./components/sentiment/MonthSelector";
import SentimentTrendLine from "./components/sentiment/SentimentTrendLine";
import RatioBarChart from "./components/sentiment/RatioBarChart";
import KpiSummaryCards from "./components/sentiment/KpiSummaryCards";
import SentimentDonutChart from "./components/sentiment/SentimentDonutChart";

export default function App() {
  const [categories, setCategories] = useState([]);
  const [dateRange, setDateRange] = useState({
    startDate: "2023-03-01",
    endDate: "2024-03-01",
  });
  const [viewMode, setViewMode] = useState("cpi");
  const [selectedMonth, setSelectedMonth] = useState("2006-01-01");

  const selectedData = categories.length > 0 ? cpiData[categories[0]] : [];

  useEffect(() => {
    setCategories([]);
  }, [viewMode]);

  const handleCategoryChange = (newCategories) => {
    if (viewMode === "category" && newCategories.length <= 1) {
      setCategories(newCategories);
    }
  };

  const sentimentDashboard = getSentimentDashboardData(
    sentimentData,
    cpiData,
    selectedMonth
  );

  return (
    <div
      style={{
        backgroundColor: "#ffffff",
        fontFamily: "Pretendard, sans-serif",
        boxSizing: "border-box",
        overflowX: "hidden",
        maxWidth: "100%",
        minHeight: "100vh",
      }}
    >
      {/* 상단 버튼 */}
      <div style={{ textAlign: "right", padding: "0.25rem 0.3rem", backgroundColor: "#f9fafb" }}>
        {["cpi", "category", "sentiment"].map((mode) => (
          <button
            key={mode}
            onClick={() => setViewMode(mode)}
            style={{
              marginRight: mode !== "sentiment" ? "0.25rem" : 0,
              backgroundColor: viewMode === mode ? "#3b82f6" : "#e5e7eb",
              color: viewMode === mode ? "#fff" : "#000",
              fontSize: "12px",
              padding: "0.3rem 3.0rem",
              border: "none",
              borderRadius: "6px",
              cursor: "pointer",
            }}
          >
            {mode === "cpi" ? "CPI 전용 차트" : mode === "category" ? "품목별 차트" : "감성 분석"}
          </button>
        ))}
      </div>

      {/* 📦 본문 + 테이블 전체를 하나로 묶음 */}
      <div
        style={{
          marginTop: "0",
          border: "1px solid #d1d5db",
          borderRadius: "12px",
          overflow: "hidden",
        }}
      >
        {/* 본문 영역 */}
        <div
          style={{
            display: "flex",
            height: viewMode === "sentiment" ? "calc(100vh - 48px)" : "auto",
          }}
        >
          {/* 왼쪽 본문 */}
          <div
            style={{
              flex: 3,
              display: "flex",
              flexDirection: "column",
              padding: "1rem",
              gap: "1rem",
              borderRight: "1px solid #d1d5db",
            }}
          >
            {viewMode === "cpi" && (
              <div style={{ height: "42%" }}>
                <CPIOnlyChart {...dateRange} />
              </div>
            )}
            {viewMode === "category" && (
              <CategoryChart selectedCategory={categories[0]} {...dateRange} />
            )}
            {viewMode === "sentiment" && (
              <>
                <SentimentTrendLine
                  months={sentimentDashboard.monthList}
                  sentiment={sentimentDashboard.sentiment_trend.scores}
                  cpiActual={sentimentDashboard.cpiActual}
                  cpiPredicted={sentimentDashboard.cpiPredicted}
                />
                <RatioBarChart
                  months={sentimentDashboard.monthList}
                  positive={sentimentDashboard.stacked_bar.positive_ratio}
                  negative={sentimentDashboard.stacked_bar.negative_ratio}
                  newsCount={sentimentDashboard.stacked_bar.news_count}
                  N_pos={sentimentDashboard.stacked_bar.N_pos}
                  N_neg={sentimentDashboard.stacked_bar.N_neg}
                />
              </>
            )}
          </div>

          {/* 오른쪽 사이드바 */}
          <div
            style={{
              flex: 1.3,
              display: "flex",
              flexDirection: "column",
              padding: "1rem",
              gap: "1rem",
              backgroundColor: "#fff",
            }}
          >
            {/* (기존 sidebar 코드 그대로 유지) */}
            {/* DateRangePicker, 예측률, 향후예측 등 */}
            {viewMode === "cpi" && (
              <>
                <DateRangePicker {...dateRange} onChange={setDateRange} />
                {/* 예측률 평균 & 향후 예측 */}
                {/* ... */}
              </>
            )}
            {viewMode === "category" && (
              <CategoryFilter
                selected={categories}
                onChange={handleCategoryChange}
                onReset={() => setCategories([])}
              />
            )}
            {viewMode === "sentiment" && sentimentDashboard.kpi_summary && (
              <>
                <MonthSelector
                  selected={selectedMonth}
                  onChange={setSelectedMonth}
                  availableMonths={sentimentData.map((d) => d.date.slice(0, 7))}
                />
                <KpiSummaryCards {...sentimentDashboard.kpi_summary} />
                <SentimentDonutChart
                  N_pos={sentimentDashboard.kpi_summary.N_pos}
                  N_neg={sentimentDashboard.kpi_summary.N_neg}
                />
              </>
            )}
          </div>
        </div>

        {/* ✅ 구분선 (위/아래 여백 동일하게) */}
        {viewMode !== "sentiment" && (
          <div
            style={{
              height: "1px",
              backgroundColor: "#e5e7eb",
              width: "100%",
            }}
          />
        )}

        {/* 하단 테이블 */}
        {viewMode !== "sentiment" && (
          <div
            style={{
              backgroundColor: "#fff",
              padding: "1rem 1rem",
              overflowX: "auto",
              boxSizing: "border-box",
              width: "100%",
            }}
          >
            {viewMode === "cpi" && <CPITableCPI {...dateRange} />}
            {viewMode === "category" && (
              <CPITableCategory category={categories[0]} {...dateRange} />
            )}
          </div>
        )}
      </div>
    </div>
  );
}
