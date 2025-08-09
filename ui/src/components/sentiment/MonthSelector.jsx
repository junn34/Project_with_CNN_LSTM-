import { useState } from "react";
import { format } from "date-fns";

export default function MonthSelector({ selectedMonth, onChange, availableMonths }) {
  const [value, setValue] = useState(selectedMonth);

  const handleChange = (e) => {
    setValue(e.target.value);
    onChange(e.target.value);
  };

  return (
    <div
      style={{
        border: "1px solid #d1d5db",
        borderRadius: "12px",
        padding: "1rem",
        backgroundColor: "#ffffff",
        fontSize: "14px",
        width: "100%",
        boxSizing: "border-box",
        marginBottom: "1rem",
      }}
    >
      <div
        style={{
          fontWeight: "600",
          marginBottom: "0.75rem",
          color: "#374151",
        }}
      >
        기준 월 선택
      </div>

      <select
        value={value}
        onChange={handleChange}
        style={{
          width: "100%",
          padding: "8px 12px",
          fontSize: "14px",
          border: "1px solid #d1d5db",
          borderRadius: "6px",
          backgroundColor: "#f9fafb",
        }}
      >
        {availableMonths.map((month) => (
          <option key={month} value={month}>
            {format(new Date(month), "yyyy-MM")}
          </option>
        ))}
      </select>
    </div>
  );
}
