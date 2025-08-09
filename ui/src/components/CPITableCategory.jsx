import React from "react";
import cpiData from "../data/parsedCPIData_with_full_predictions.json";

const CPITableCategory = ({ category, startDate, endDate }) => {
  // 날짜 범위만 기준으로 헤더를 항상 생성
  const referenceCategory = Object.keys(cpiData)[0];
  const dates = cpiData[referenceCategory]
    .filter((d) => d.날짜 >= startDate && d.날짜 <= endDate)
    .map((d) => d.날짜);

  // 초기값은 모두 "-"
  let actualRow = dates.map(() => "-");
  let predictedRow = dates.map(() => "-");
  let rateRow = dates.map(() => "-");

  // category가 있을 경우 실제 데이터로 채움
  if (category && cpiData[category]) {
    const records = cpiData[category].filter(
      (r) => r.날짜 >= startDate && r.날짜 <= endDate
    );

    const getRecordByDate = (date) => records.find((r) => r.날짜 === date);

    actualRow = dates.map((date) => {
      const r = getRecordByDate(date);
      return r?.값 !== undefined ? r.값.toFixed(2) : "-";
    });

    predictedRow = dates.map((date) => {
      const r = getRecordByDate(date);
      return r?.예측값 !== undefined ? r.예측값.toFixed(2) : "-";
    });

    rateRow = dates.map((date) => {
      const r = getRecordByDate(date);
      if (
        typeof r?.값 === "number" &&
        typeof r?.예측값 === "number" &&
        r.값 !== 0
      ) {
        const rate = ((r.예측값 - r.값) / r.값) * 100;
        return `${rate.toFixed(1)}%`;
      }
      return "-";
    });
  }

  return (
    <div
      style={{
        border: "1px solid #d1d5db",
        borderRadius: 8,
        padding: "1rem",
        backgroundColor: "#fff",
      }}
    >
      <h3 style={{ fontWeight: 700, fontSize: "16px", marginBottom: "12px" }}>
        {category || "📌 항목을 선택해주세요"}
      </h3>
      <div style={{ overflowX: "auto", width: "100%" }}>
        <table
          style={{
            tableLayout: "fixed",
            borderCollapse: "collapse",
            minWidth: `${dates.length * 140 + 150}px`,
            fontSize: "12px",
          }}
        >
          <thead>
            <tr style={{ backgroundColor: "#f1f5f9" }}>
              <th
                style={{
                  width: "150px",
                  padding: "4px 6px",
                  textAlign: "left",
                  whiteSpace: "nowrap",
                }}
              >
                항목 / 날짜
              </th>
              {dates.map((d) => (
                <th
                  key={d}
                  style={{
                    width: "140px",
                    padding: "4px 6px",
                    textAlign: "center",
                    whiteSpace: "nowrap",
                  }}
                >
                  {d}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {[["실제값", actualRow], ["예측값", predictedRow], ["예측률", rateRow]].map(
              ([label, row]) => (
                <tr key={label}>
                  <td
                    style={{
                      padding: "4px 6px",
                      fontWeight: 500,
                      whiteSpace: "nowrap",
                    }}
                  >
                    {label}
                  </td>
                  {row.map((v, i) => (
                    <td
                      key={i}
                      style={{
                        padding: "4px 6px",
                        textAlign: "center",
                        whiteSpace: "nowrap",
                      }}
                    >
                      {v}
                    </td>
                  ))}
                </tr>
              )
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default CPITableCategory;
