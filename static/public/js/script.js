function updateAttendanceInfo() {
  fetch("/attendance_info")
    .then((response) => response.json())
    .then((data) => {
      const attendanceInfoDiv = document.getElementById("attendance_info");
      attendanceInfoDiv.innerHTML = "<h3>Attendance Information:</h3>";
      const table = document.createElement("table");
      table.style.borderCollapse = "collapse";
      table.style.width = "100%";

      const headerRow = table.insertRow(0);
      const headers = ["Name", "Timestamp", "Present", "Actions"];
      headers.forEach((headerText) => {
        const header = document.createElement("th");
        header.textContent = headerText;
        header.style.border = "1px solid #dddddd";
        header.style.padding = "8px";
        header.style.textAlign = "left";
        headerRow.appendChild(header);
      });

      const uniqueNames = new Set();

      data.forEach((attendee) => {
        // Check if the name is not already in the set
        if (!uniqueNames.has(attendee.name)) {
          const row = table.insertRow(-1);

          // Cell 1: Name
          const cell1 = row.insertCell(0);
          cell1.textContent = attendee.name;
          cell1.style.border = "1px solid #dddddd";
          cell1.style.padding = "8px";

          // Cell 2: Timestamp
          const cell2 = row.insertCell(1);
          cell2.textContent = attendee.timestamp;
          cell2.style.border = "1px solid #dddddd";
          cell2.style.padding = "8px";

          // Cell 3: Present
          const cell3 = row.insertCell(2);
          cell3.textContent = attendee.present;
          cell3.style.border = "1px solid #dddddd";
          cell3.style.padding = "8px";

          // Cell 4: Actions (Remove button)
          const cell4 = row.insertCell(3);
          const removeButton = document.createElement("button");
          removeButton.textContent = "Remove";
          removeButton.onclick = function () {
            // Call the Flask route to remove the attendee
            removeAttendee(attendee.name);
          };
          cell4.appendChild(removeButton);
          cell4.style.border = "1px solid #dddddd";
          cell4.style.padding = "8px";
          cell4.style.textAlign = "center";

          // Add the name to the set to track duplicates
          uniqueNames.add(attendee.name);
        }
      });

      attendanceInfoDiv.appendChild(table);
    })
    .catch((error) =>
      console.error("Error fetching attendance information:", error)
    );
}

// Function to remove an attendee
function removeAttendee(name) {
  // Call the Flask route to remove the attendee
  fetch("/remove_attendee", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ name: name }),
  })
    .then((response) => response.text())
    .then((message) => {
      console.log(message);
      // Refresh the attendance information after removal
      updateAttendanceInfo();
    })
    .catch((error) => {
      console.error("Error removing attendee:", error);
    });
}


// Update attendance information every 5 seconds
setInterval(updateAttendanceInfo, 10000);

// Function to start attendance
function startAttendance() {
  // Create an img element
  const imgElement = document.createElement("img");

  // Set attributes for the img element
  imgElement.id = "video_feed";
  imgElement.src = "video_feed";
  imgElement.alt = "Video Feed";
  imgElement.style.width = "100%";
  imgElement.style.height = "100%";
  const videoContainer = document.getElementById("video_stream"); //
  videoContainer.innerHTML = "";
  videoContainer.appendChild(imgElement);

  fetch("/startAttendance", {
    method: "POST",
  })
    .then((response) => response.text())
    .then((message) => {
      console.log(message);
    })
    .catch((error) => {
      console.error("Error saving attendance:", error);
    });

  console.log("Attendance started!");
}

// Function to save attendance
function saveAttendance() {
  // Call the Flask route to save attendance
  fetch("/save_attendance", {
    method: "POST",
  })
    .then((response) => response.text())
    .then((message) => {
      console.log(message);
    })
    .catch((error) => {
      console.error("Error saving attendance:", error);
    });
}

// Function to stop the video feed
function stopVideoFeed() {
  // Call the Flask route to stop the video feed
  fetch("/stopVideoFeed", {
    method: "POST",
  })
    .then((response) => response.text())
    .then((message) => {
      console.log(message);
    })
    .catch((error) => {
      console.error("Error stopping video feed:", error);
    });
}

// Function to clear attendance
function clearAttendance() {
  fetch("/clearAttendance", {
    method: "POST",
  })
    .then((response) => response.text())
    .then((message) => {
      console.log(message);
      // Update attendance information after clearing
      updateAttendanceInfo();
    })
    .catch((error) => {
      console.error("Error clearing attendance:", error);
    });
}
